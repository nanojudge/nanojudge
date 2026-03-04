/// Gaussian Bradley-Terry MCMC sampler.
///
/// Uses logit(P) as direct observation of strength difference.
/// Metropolis-Hastings within Gibbs sampling for posterior inference.
/// Internal module — operates on pre-mapped `usize` indices, not caller IDs.
use rand::Rng;

use crate::types::{IndexedComparison, RankedItem};

/// Internal representation of a comparison in logit space.
struct LogitComparison {
    idx1: usize,
    idx2: usize,
    logit_y: f64,
}

/// Result from `calculate_with_samples` and `calculate_incremental_with_samples`.
pub struct SamplesResult {
    pub sorted_samples: Vec<Vec<f64>>,
    pub means: Vec<f64>,
    pub top_k_probs: Option<Vec<f64>>,
    /// Estimated positional bias in logit space (0.0 = no bias).
    pub bias_logit_mean: f64,
    /// Sorted bias samples in logit space (for confidence interval computation).
    pub bias_logit_samples: Vec<f64>,
}

pub struct GaussianBT {
    /// Number of real items (excluding ghost).
    num_items: usize,
    /// Total items including ghost.
    total: usize,
    /// Ghost player index.
    ghost_idx: usize,
    /// Comparisons in logit space (raw, not de-biased).
    comparisons: Vec<LogitComparison>,
    /// Adjacency list: item_idx -> indices into `comparisons`.
    item_comparisons: Vec<Vec<usize>>,
    /// Current log-strengths (theta values).
    log_strengths: Vec<f64>,
    /// Regularization strength for ghost player.
    regularization_strength: f64,
    /// Number of real (non-ghost) comparisons.
    num_real_comparisons: usize,

    /// Current positional bias estimate (logit space, 0 = no bias).
    bias: f64,

    // Hyperparameters (fixed)
    prior_mu: f64,
    prior_tau2: f64,
    sigma2: f64,
    proposal_std: f64,
    bias_prior_tau2: f64,
    bias_proposal_std: f64,
}

impl GaussianBT {
    pub fn new(
        num_items: usize,
        results: &[IndexedComparison],
        regularization_strength: f64,
    ) -> Self {
        let ghost_idx = num_items;
        let total = num_items + 1;
        let prior_mu = 0.0;

        // Build comparisons — store raw logits, bias is estimated jointly
        let mut comparisons = Vec::new();
        let mut item_comparisons: Vec<Vec<usize>> = (0..total).map(|_| Vec::new()).collect();

        for &(idx1, idx2, prob) in results {
            assert!(idx1 < num_items, "item1 index {} out of range (num_items = {})", idx1, num_items);
            assert!(idx2 < num_items, "item2 index {} out of range (num_items = {})", idx2, num_items);

            // Clamp to avoid infinity — raw logit, no de-biasing
            let clamped_p = prob.clamp(0.001, 0.999);
            let logit_y = (clamped_p / (1.0 - clamped_p)).ln();

            let comp_idx = comparisons.len();
            comparisons.push(LogitComparison {
                idx1,
                idx2,
                logit_y,
            });
            item_comparisons[idx1].push(comp_idx);
            item_comparisons[idx2].push(comp_idx);
        }

        let num_real_comparisons = comparisons.len();

        // Ghost regularization comparisons
        if regularization_strength > 0.0 {
            for i in 0..num_items {
                let comp_idx = comparisons.len();
                comparisons.push(LogitComparison {
                    idx1: i,
                    idx2: ghost_idx,
                    logit_y: 0.0,
                });
                item_comparisons[i].push(comp_idx);
                item_comparisons[ghost_idx].push(comp_idx);
            }
        }

        GaussianBT {
            num_items,
            total,
            ghost_idx,
            comparisons,
            item_comparisons,
            log_strengths: vec![prior_mu; total],
            regularization_strength,
            num_real_comparisons,
            bias: 0.0, // start at no bias, let MCMC estimate it
            prior_mu,
            prior_tau2: 10.0,
            sigma2: 1.0,
            proposal_std: 0.3,
            bias_prior_tau2: 2.0, // N(0, √2) in logit space — 95% mass covers ~0.06 to ~0.94 in probability
            bias_proposal_std: 0.15,
        }
    }

    fn log_posterior(&self, item_idx: usize, log_strength: f64) -> f64 {
        let prior_diff = log_strength - self.prior_mu;
        let mut log_prob = -0.5 * prior_diff * prior_diff / self.prior_tau2;

        for &comp_idx in &self.item_comparisons[item_idx] {
            let comp = &self.comparisons[comp_idx];

            let is_ghost = comp.idx1 == self.ghost_idx || comp.idx2 == self.ghost_idx;

            let strength_diff = if comp.idx1 == item_idx {
                log_strength - self.log_strengths[comp.idx2]
            } else {
                self.log_strengths[comp.idx1] - log_strength
            };

            // Bias only applies to real comparisons, not ghost regularization
            let predicted = if is_ghost { strength_diff } else { strength_diff + self.bias };

            let residual = comp.logit_y - predicted;

            let effective_sigma2 = if is_ghost {
                self.sigma2 / self.regularization_strength
            } else {
                self.sigma2
            };

            log_prob += -0.5 * residual * residual / effective_sigma2;
        }

        log_prob
    }

    fn update_strength(&mut self, item_idx: usize, rng: &mut impl Rng) {
        let current = self.log_strengths[item_idx];
        let proposed = current + (rng.random::<f64>() - 0.5) * 2.0 * self.proposal_std;

        let log_posterior_current = self.log_posterior(item_idx, current);
        let log_posterior_proposed = self.log_posterior(item_idx, proposed);

        if rng.random::<f64>().ln() < (log_posterior_proposed - log_posterior_current) {
            self.log_strengths[item_idx] = proposed;
        }
    }

    /// Log-posterior for the positional bias parameter.
    /// Iterates over real comparisons only (not ghost).
    fn log_posterior_bias(&self, bias: f64) -> f64 {
        // Prior: N(0, bias_prior_tau2)
        let mut log_prob = -0.5 * bias * bias / self.bias_prior_tau2;

        // Likelihood over real comparisons only
        for comp in &self.comparisons[..self.num_real_comparisons] {
            let predicted = self.log_strengths[comp.idx1] - self.log_strengths[comp.idx2] + bias;
            let residual = comp.logit_y - predicted;
            log_prob += -0.5 * residual * residual / self.sigma2;
        }

        log_prob
    }

    fn update_bias(&mut self, rng: &mut impl Rng) {
        let current = self.bias;
        let proposed = current + (rng.random::<f64>() - 0.5) * 2.0 * self.bias_proposal_std;

        let log_post_current = self.log_posterior_bias(current);
        let log_post_proposed = self.log_posterior_bias(proposed);

        if rng.random::<f64>().ln() < (log_post_proposed - log_post_current) {
            self.bias = proposed;
        }
    }

    fn normalize_log_strengths(&mut self) {
        let mean = self.log_strengths.iter().sum::<f64>() / self.total as f64;
        for val in &mut self.log_strengths {
            *val -= mean;
        }
    }

    fn gibbs_iteration(&mut self, rng: &mut impl Rng) {
        for i in 0..self.total {
            self.update_strength(i, rng);
        }
        self.update_bias(rng);
    }

    /// Main MCMC calculation returning ranked results.
    /// Items in the returned RankedItem use index-as-i64 (caller maps to real IDs).
    pub fn calculate(
        &mut self,
        mcmc_iterations: usize,
        confidence_level: f64,
        burn_in: usize,
    ) -> Vec<RankedItem> {
        let mut rng = rand::rng();
        let n = self.num_items;

        for _ in 0..burn_in {
            self.gibbs_iteration(&mut rng);
            self.normalize_log_strengths();
        }

        let mut samples_per_item: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(mcmc_iterations)).collect();

        for _ in 0..mcmc_iterations {
            self.gibbs_iteration(&mut rng);
            self.normalize_log_strengths();
            for idx in 0..n {
                samples_per_item[idx].push(self.log_strengths[idx].exp());
            }
        }

        let alpha = 1.0 - confidence_level;
        let mut results = Vec::with_capacity(n);

        for idx in 0..n {
            let samples = &mut samples_per_item[idx];
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let lower_idx = ((alpha / 2.0) * samples.len() as f64).floor() as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * samples.len() as f64).floor() as usize;
            let upper_idx = upper_idx.saturating_sub(1).max(lower_idx);

            results.push(RankedItem {
                item: idx as i64,
                score: mean,
                lower_bound: samples[lower_idx],
                upper_bound: samples[upper_idx],
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get the current bias estimate converted to probability space.
    pub fn estimated_bias_probability(&self) -> f64 {
        1.0 / (1.0 + (-self.bias).exp())
    }

    /// Run MCMC sampling loop and collect results. Shared by cold-start and warm-start paths.
    fn collect_samples(
        &mut self,
        iterations: usize,
        top_k: usize,
        rng: &mut impl Rng,
    ) -> SamplesResult {
        let n = self.num_items;
        let effective_k = top_k.min(n);
        let mut top_k_count: Option<Vec<usize>> = if top_k > 0 { Some(vec![0; n]) } else { None };
        let mut sort_indices: Vec<usize> = (0..n).collect();

        let mut samples_per_item: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(iterations)).collect();
        let mut bias_samples = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            self.gibbs_iteration(rng);
            self.normalize_log_strengths();

            for idx in 0..n {
                samples_per_item[idx].push(self.log_strengths[idx].exp());
            }
            bias_samples.push(self.bias);

            if let Some(ref mut counts) = top_k_count {
                for j in 0..n { sort_indices[j] = j; }
                sort_indices.sort_by(|&a, &b| {
                    self.log_strengths[b].partial_cmp(&self.log_strengths[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                for k in 0..effective_k {
                    counts[sort_indices[k]] += 1;
                }
            }
        }

        let mut sorted_samples = Vec::with_capacity(n);
        let mut means = Vec::with_capacity(n);

        for idx in 0..n {
            let samples = &mut samples_per_item[idx];
            means.push(samples.iter().sum::<f64>() / samples.len() as f64);
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted_samples.push(std::mem::take(samples));
        }

        let bias_logit_mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
        bias_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        SamplesResult {
            sorted_samples,
            means,
            top_k_probs: top_k_count.map(|c| c.iter().map(|&v| v as f64 / iterations as f64).collect()),
            bias_logit_mean,
            bias_logit_samples: bias_samples,
        }
    }

    /// Cold-start MCMC returning raw sorted samples.
    pub fn calculate_with_samples(
        &mut self,
        mcmc_iterations: usize,
        burn_in: usize,
        top_k: usize,
    ) -> SamplesResult {
        let mut rng = rand::rng();

        for _ in 0..burn_in {
            self.gibbs_iteration(&mut rng);
            self.normalize_log_strengths();
        }

        self.collect_samples(mcmc_iterations, top_k, &mut rng)
    }

    /// Get current state for warm-starting (exp of log-strengths for real items).
    pub fn get_current_state(&self) -> Vec<f64> {
        self.log_strengths[..self.num_items].iter().map(|&v| v.exp()).collect()
    }

    /// Warm-start MCMC returning raw sorted samples.
    pub fn calculate_incremental_with_samples(
        &mut self,
        previous_lambda: &[f64],
        new_iterations: usize,
        burn_in: usize,
        top_k: usize,
    ) -> SamplesResult {
        let n = self.num_items;
        assert_eq!(previous_lambda.len(), n, "Previous state size mismatch");

        for i in 0..n {
            self.log_strengths[i] = previous_lambda[i].ln();
        }
        self.log_strengths[self.ghost_idx] = 0.0;

        let mut rng = rand::rng();

        for _ in 0..burn_in {
            self.gibbs_iteration(&mut rng);
            self.normalize_log_strengths();
        }

        self.collect_samples(new_iterations, top_k, &mut rng)
    }

    /// Compute confidence intervals from pre-sorted MCMC samples.
    /// Items in the returned RankedItem use index-as-i64 (caller maps to real IDs).
    pub fn compute_confidence_intervals_from_sorted_samples(
        sorted_samples: &[Vec<f64>],
        means: &[f64],
        confidence_level: f64,
    ) -> Vec<RankedItem> {
        let alpha = 1.0 - confidence_level;
        let num_items = sorted_samples.len();
        let mut results = Vec::with_capacity(num_items);

        for i in 0..num_items {
            let samples = &sorted_samples[i];
            let n = samples.len();

            if n == 0 {
                results.push(RankedItem {
                    item: i as i64,
                    score: means[i],
                    lower_bound: means[i],
                    upper_bound: means[i],
                });
                continue;
            }

            let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * n as f64).floor() as usize;
            let upper_idx = upper_idx.saturating_sub(1).max(lower_idx);

            results.push(RankedItem {
                item: i as i64,
                score: means[i],
                lower_bound: samples[lower_idx],
                upper_bound: samples[upper_idx],
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn num_items(&self) -> usize {
        self.num_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(i1: usize, i2: usize, prob: f64) -> [IndexedComparison; 2] {
        [(i1, i2, prob), (i2, i1, 1.0 - prob)]
    }

    #[test]
    fn test_basic_mcmc_ranking() {
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(0, 2, 0.8),
            make_pair(1, 2, 0.7),
        ].into_iter().flatten().collect();

        let mut mcmc = GaussianBT::new(3, &results, 0.01);
        let ranked = mcmc.calculate(500, 0.95, 200);

        assert_eq!(ranked[0].item, 0); // A first
        assert_eq!(ranked[2].item, 2); // C last
    }

    #[test]
    fn test_warm_start() {
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(1, 2, 0.8),
        ].into_iter().flatten().collect();

        let mut mcmc = GaussianBT::new(3, &results, 0.01);
        let _result1 = mcmc.calculate_with_samples(50, 50, 0);
        let state = mcmc.get_current_state();

        let mut mcmc2 = GaussianBT::new(3, &results, 0.01);
        let result2 = mcmc2.calculate_incremental_with_samples(&state, 50, 0, 0);

        assert_eq!(result2.means.len(), 3);
    }

    #[test]
    fn test_top_k_probs() {
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(0, 2, 0.9),
            make_pair(0, 3, 0.9),
            make_pair(1, 2, 0.7),
            make_pair(1, 3, 0.7),
            make_pair(2, 3, 0.6),
        ].into_iter().flatten().collect();

        let mut mcmc = GaussianBT::new(4, &results, 0.01);
        let result = mcmc.calculate_with_samples(200, 100, 2);

        let probs = result.top_k_probs.unwrap();
        assert_eq!(probs.len(), 4);
        // Item 0 (strongest) should have highest P(top 2)
        assert!(probs[0] > probs[3], "Item 0 should have higher P(top K) than item 3");
    }

    #[test]
    fn test_compute_confidence_intervals_from_sorted_samples() {
        let means = vec![2.0, 1.0];
        let sorted_samples = vec![
            vec![1.0, 1.5, 2.0, 2.5, 3.0],
            vec![0.5, 0.8, 1.0, 1.2, 1.5],
        ];

        let results = GaussianBT::compute_confidence_intervals_from_sorted_samples(&sorted_samples, &means, 0.90);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].item, 0); // Higher score first
        assert!(results[0].lower_bound <= results[0].score);
        assert!(results[0].upper_bound >= results[0].score);
    }
}
