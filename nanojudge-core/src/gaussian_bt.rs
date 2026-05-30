/// Gaussian Bradley-Terry MCMC sampler with multi-judge support.
///
/// Uses logit(P) as direct observation of strength difference.
/// Metropolis-Hastings within Gibbs sampling for posterior inference.
///
/// Two modes:
/// - **Logprobs mode**: Per-judge decisiveness D_k and positional bias γ_k.
/// - **No-logprobs mode**: Per-judge positional bias γ_k only, no D_k.
///
/// Internal module — operates on pre-mapped `usize` indices, not caller IDs.
use std::collections::HashMap;
use rand::Rng;

use crate::types::{IndexedComparison, JudgeInfo, RankedItem, ScoringOptions};

/// Internal representation of a comparison in logit space.
struct LogitComparison {
    idx1: usize,
    idx2: usize,
    logit_y: f64,
    /// Internal judge index (into biases/decisiveness vecs).
    judge_idx: usize,
}

/// Result from `calculate_with_samples` and `calculate_incremental_with_samples`.
pub struct SamplesResult {
    pub sorted_samples: Vec<Vec<f64>>,
    pub means: Vec<f64>,
    pub top_k_probs: Option<Vec<f64>>,
    /// Per-judge bias samples in logit space. Outer vec indexed by judge.
    pub bias_logit_samples: Vec<Vec<f64>>,
    /// Per-judge bias means in logit space.
    pub bias_logit_means: Vec<f64>,
    /// Per-judge decisiveness samples in log space (ln(D_k)). Empty vecs in no-logprobs mode.
    pub log_decisiveness_samples: Vec<Vec<f64>>,
    /// Per-judge decisiveness means in log space. Empty in no-logprobs mode.
    pub log_decisiveness_means: Vec<f64>,
    /// Number of comparisons per judge.
    pub comparisons_per_judge: Vec<usize>,
}

pub struct GaussianBT {
    /// Number of real items (excluding ghost).
    num_items: usize,
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

    /// Number of judges.
    num_judges: usize,
    /// Whether we're in logprobs mode (with decisiveness).
    logprobs_mode: bool,
    /// Per-judge positional bias (logit space). Length = num_judges.
    biases: Vec<f64>,
    /// Per-judge log-decisiveness (ln(D_k)). Empty in no-logprobs mode.
    decisiveness: Vec<f64>,
    /// Number of comparisons per judge (real only, not ghost).
    comparisons_per_judge: Vec<usize>,

    /// Prior mean for positional bias (logit space).
    bias_prior_mu: f64,

    // Hyperparameters (fixed)
    prior_mu: f64,
    prior_tau2: f64,
    sigma2: f64,
    proposal_std: f64,
    bias_prior_tau2: f64,
    bias_proposal_std: f64,
    decisiveness_prior_tau2: f64,
    decisiveness_proposal_std: f64,
}

impl GaussianBT {
    pub fn new(
        num_items: usize,
        results: &[IndexedComparison],
        options: &ScoringOptions,
        judge_info: &JudgeInfo,
    ) -> Self {
        let ghost_idx = num_items;
        let total = num_items + 1;
        let prior_mu = 0.0;
        let num_judges = judge_info.judge_ids.len();

        // Build comparisons — store raw logits, bias is estimated jointly
        let mut comparisons = Vec::new();
        let mut item_comparisons: Vec<Vec<usize>> = (0..total).map(|_| Vec::new()).collect();
        let mut comparisons_per_judge = vec![0usize; num_judges];

        for &(idx1, idx2, prob, judge_idx) in results {
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
                judge_idx,
            });
            item_comparisons[idx1].push(comp_idx);
            item_comparisons[idx2].push(comp_idx);
            comparisons_per_judge[judge_idx] += 1;
        }

        let num_real_comparisons = comparisons.len();

        // Ghost regularization comparisons — use judge_idx 0 but it doesn't matter
        // because ghost comparisons are exempt from judge parameters
        if options.regularization_strength > 0.0 {
            for i in 0..num_items {
                let comp_idx = comparisons.len();
                comparisons.push(LogitComparison {
                    idx1: i,
                    idx2: ghost_idx,
                    logit_y: 0.0,
                    judge_idx: 0, // unused for ghost
                });
                item_comparisons[i].push(comp_idx);
                item_comparisons[ghost_idx].push(comp_idx);
            }
        }

        // Initialize per-judge parameters
        let biases = vec![options.bias_prior_logit; num_judges];
        let decisiveness = if judge_info.logprobs_mode {
            vec![0.0; num_judges] // ln(1.0) = 0.0
        } else {
            Vec::new()
        };

        GaussianBT {
            num_items,
            ghost_idx,
            comparisons,
            item_comparisons,
            log_strengths: vec![prior_mu; total],
            regularization_strength: options.regularization_strength,
            num_real_comparisons,
            num_judges,
            logprobs_mode: judge_info.logprobs_mode,
            biases,
            decisiveness,
            comparisons_per_judge,
            bias_prior_mu: options.bias_prior_logit,
            prior_mu,
            prior_tau2: options.prior_tau2,
            sigma2: options.sigma2,
            proposal_std: options.proposal_std,
            bias_prior_tau2: options.bias_prior_tau2,
            bias_proposal_std: options.bias_proposal_std,
            decisiveness_prior_tau2: options.decisiveness_prior_tau2,
            decisiveness_proposal_std: options.decisiveness_proposal_std,
        }
    }

    /// Compute the predicted logit value using a hypothetical strength for one item.
    fn predicted_logit_with_strength(&self, comp: &LogitComparison, is_ghost: bool, item_idx: usize, log_strength: f64) -> f64 {
        let strength_diff = if comp.idx1 == item_idx {
            log_strength - self.log_strengths[comp.idx2]
        } else {
            self.log_strengths[comp.idx1] - log_strength
        };

        if is_ghost {
            strength_diff
        } else if self.logprobs_mode {
            let d_k = self.decisiveness[comp.judge_idx].exp();
            let gamma_k = self.biases[comp.judge_idx];
            d_k * strength_diff + gamma_k
        } else {
            let gamma_k = self.biases[comp.judge_idx];
            strength_diff + gamma_k
        }
    }

    fn log_posterior(&self, item_idx: usize, log_strength: f64) -> f64 {
        let prior_diff = log_strength - self.prior_mu;
        let mut log_prob = -0.5 * prior_diff * prior_diff / self.prior_tau2;

        for &comp_idx in &self.item_comparisons[item_idx] {
            let comp = &self.comparisons[comp_idx];
            let is_ghost = comp.idx1 == self.ghost_idx || comp.idx2 == self.ghost_idx;

            let predicted = self.predicted_logit_with_strength(comp, is_ghost, item_idx, log_strength);
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

    /// Log-posterior for a judge's positional bias parameter.
    /// Iterates over real comparisons only (not ghost).
    fn log_posterior_bias(&self, judge_idx: usize, bias: f64) -> f64 {
        let bias_diff = bias - self.bias_prior_mu;
        let mut log_prob = -0.5 * bias_diff * bias_diff / self.bias_prior_tau2;

        for comp in &self.comparisons[..self.num_real_comparisons] {
            if comp.judge_idx != judge_idx {
                continue;
            }

            let strength_diff = self.log_strengths[comp.idx1] - self.log_strengths[comp.idx2];
            let predicted = if self.logprobs_mode {
                let d_k = self.decisiveness[judge_idx].exp();
                d_k * strength_diff + bias
            } else {
                strength_diff + bias
            };
            let residual = comp.logit_y - predicted;
            log_prob += -0.5 * residual * residual / self.sigma2;
        }

        log_prob
    }

    fn update_bias(&mut self, judge_idx: usize, rng: &mut impl Rng) {
        let current = self.biases[judge_idx];
        let proposed = current + (rng.random::<f64>() - 0.5) * 2.0 * self.bias_proposal_std;

        let log_post_current = self.log_posterior_bias(judge_idx, current);
        let log_post_proposed = self.log_posterior_bias(judge_idx, proposed);

        if rng.random::<f64>().ln() < (log_post_proposed - log_post_current) {
            self.biases[judge_idx] = proposed;
        }
    }

    /// Log-posterior for a judge's log-decisiveness parameter (logprobs mode only).
    fn log_posterior_decisiveness(&self, judge_idx: usize, log_d: f64) -> f64 {
        // Prior: N(0, decisiveness_prior_tau2) in log space
        let mut log_prob = -0.5 * log_d * log_d / self.decisiveness_prior_tau2;

        let d_k = log_d.exp();

        for comp in &self.comparisons[..self.num_real_comparisons] {
            if comp.judge_idx != judge_idx {
                continue;
            }

            let strength_diff = self.log_strengths[comp.idx1] - self.log_strengths[comp.idx2];
            let predicted = d_k * strength_diff + self.biases[judge_idx];
            let residual = comp.logit_y - predicted;
            log_prob += -0.5 * residual * residual / self.sigma2;
        }

        log_prob
    }

    fn update_decisiveness(&mut self, judge_idx: usize, rng: &mut impl Rng) {
        let current = self.decisiveness[judge_idx];
        let proposed = current + (rng.random::<f64>() - 0.5) * 2.0 * self.decisiveness_proposal_std;

        let log_post_current = self.log_posterior_decisiveness(judge_idx, current);
        let log_post_proposed = self.log_posterior_decisiveness(judge_idx, proposed);

        if rng.random::<f64>().ln() < (log_post_proposed - log_post_current) {
            self.decisiveness[judge_idx] = proposed;
        }
    }

    /// Normalize decisiveness: subtract mean of ln(D_k) values to enforce ∏ D_k = 1.
    fn normalize_decisiveness(&mut self) {
        if self.decisiveness.is_empty() {
            return;
        }
        let mean = self.decisiveness.iter().sum::<f64>() / self.decisiveness.len() as f64;
        for val in &mut self.decisiveness {
            *val -= mean;
        }
    }

    fn normalize_log_strengths(&mut self) {
        // Ghost is a fixed anchor at 0 — exclude it from the mean and leave it untouched.
        let mean = self.log_strengths[..self.num_items].iter().sum::<f64>() / self.num_items as f64;
        for val in &mut self.log_strengths[..self.num_items] {
            *val -= mean;
        }
        self.log_strengths[self.ghost_idx] = 0.0;
    }

    fn gibbs_iteration(&mut self, rng: &mut impl Rng) {
        // Step 1: Update each item's log-strength (ghost is a fixed anchor, skip it)
        for i in 0..self.num_items {
            self.update_strength(i, rng);
        }

        // Step 2: Update each judge's positional bias
        for k in 0..self.num_judges {
            self.update_bias(k, rng);
        }

        // Step 3 (logprobs mode only): Update each judge's decisiveness
        if self.logprobs_mode {
            for k in 0..self.num_judges {
                self.update_decisiveness(k, rng);
            }
            // Step 4: Normalize decisiveness (geometric mean anchor)
            self.normalize_decisiveness();
        }

        // Step 5: Normalize log-strengths (done by collect_samples after each iteration)
    }

    /// Run MCMC sampling loop and collect results. Shared by cold-start and warm-start paths.
    fn collect_samples(
        &mut self,
        iterations: usize,
        top_k: usize,
        rng: &mut impl Rng,
    ) -> SamplesResult {
        let n = self.num_items;
        let k = self.num_judges;
        let effective_k = top_k.min(n);
        let mut top_k_count: Option<Vec<usize>> = if top_k > 0 { Some(vec![0; n]) } else { None };
        let mut sort_indices: Vec<usize> = (0..n).collect();

        let mut samples_per_item: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(iterations)).collect();
        let mut bias_samples: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(iterations)).collect();
        let mut log_d_samples: Vec<Vec<f64>> = if self.logprobs_mode {
            (0..k).map(|_| Vec::with_capacity(iterations)).collect()
        } else {
            Vec::new()
        };

        for _ in 0..iterations {
            self.gibbs_iteration(rng);
            self.normalize_log_strengths();

            for idx in 0..n {
                samples_per_item[idx].push(self.log_strengths[idx]);
            }

            for j in 0..k {
                bias_samples[j].push(self.biases[j]);
            }

            if self.logprobs_mode {
                for j in 0..k {
                    log_d_samples[j].push(self.decisiveness[j]);
                }
            }

            if let Some(ref mut counts) = top_k_count {
                for j in 0..n { sort_indices[j] = j; }
                sort_indices.sort_by(|&a, &b| {
                    self.log_strengths[b].partial_cmp(&self.log_strengths[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                for idx in 0..effective_k {
                    counts[sort_indices[idx]] += 1;
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

        // Compute per-judge bias means and sort samples
        let mut bias_logit_means = Vec::with_capacity(k);
        for j in 0..k {
            bias_logit_means.push(bias_samples[j].iter().sum::<f64>() / bias_samples[j].len() as f64);
            bias_samples[j].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Compute per-judge decisiveness means and sort samples (logprobs mode only)
        let mut log_d_means = Vec::new();
        if self.logprobs_mode {
            for j in 0..k {
                log_d_means.push(log_d_samples[j].iter().sum::<f64>() / log_d_samples[j].len() as f64);
                log_d_samples[j].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        SamplesResult {
            sorted_samples,
            means,
            top_k_probs: top_k_count.map(|c| c.iter().map(|&v| v as f64 / iterations as f64).collect()),
            bias_logit_samples: bias_samples,
            bias_logit_means,
            log_decisiveness_samples: log_d_samples,
            log_decisiveness_means: log_d_means,
            comparisons_per_judge: self.comparisons_per_judge.clone(),
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

    /// Get current item state for warm-starting (log-strengths for real items).
    pub fn get_current_state(&self) -> Vec<f64> {
        self.log_strengths[..self.num_items].to_vec()
    }

    /// Get current per-judge biases (keyed by judge_id from the info).
    pub fn get_current_biases(&self, judge_info: &JudgeInfo) -> Vec<(u64, f64)> {
        judge_info.judge_ids.iter().enumerate()
            .map(|(idx, &id)| (id, self.biases[idx]))
            .collect()
    }

    /// Get current per-judge log-decisiveness (keyed by judge_id). Empty in no-logprobs mode.
    pub fn get_current_log_decisiveness(&self, judge_info: &JudgeInfo) -> Vec<(u64, f64)> {
        if !self.logprobs_mode {
            return Vec::new();
        }
        judge_info.judge_ids.iter().enumerate()
            .map(|(idx, &id)| (id, self.decisiveness[idx]))
            .collect()
    }

    /// Warm-start MCMC returning raw sorted samples.
    pub fn calculate_incremental_with_samples(
        &mut self,
        previous_strengths: &[f64],
        previous_biases: &[(u64, f64)],
        previous_log_decisiveness: &[(u64, f64)],
        judge_id_to_idx: &HashMap<u64, usize>,
        new_iterations: usize,
        burn_in: usize,
        top_k: usize,
    ) -> SamplesResult {
        let n = self.num_items;
        assert_eq!(previous_strengths.len(), n, "Previous state size mismatch");

        // Restore item strengths
        for i in 0..n {
            self.log_strengths[i] = previous_strengths[i];
        }
        self.log_strengths[self.ghost_idx] = 0.0;

        // Restore per-judge biases
        for &(judge_id, bias) in previous_biases {
            if let Some(&idx) = judge_id_to_idx.get(&judge_id) {
                self.biases[idx] = bias;
            }
        }

        // Restore per-judge decisiveness (logprobs mode only)
        if self.logprobs_mode {
            for &(judge_id, log_d) in previous_log_decisiveness {
                if let Some(&idx) = judge_id_to_idx.get(&judge_id) {
                    self.decisiveness[idx] = log_d;
                }
            }
        }

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
    use crate::types::ScoringOptions;

    fn single_judge_info() -> JudgeInfo {
        JudgeInfo {
            judge_ids: vec![0],
            logprobs_mode: true,
        }
    }

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(i1: usize, i2: usize, prob: f64) -> [IndexedComparison; 2] {
        [(i1, i2, prob, 0), (i2, i1, 1.0 - prob, 0)]
    }

    fn default_options() -> ScoringOptions {
        ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            sigma2: 1.0,
            proposal_std: 0.3,
            bias_prior_tau2: 2.0,
            bias_proposal_std: 0.15,
            bias_prior_logit: 0.0,
            decisiveness_prior_tau2: 1.0,
            decisiveness_proposal_std: 0.1,
        }
    }

    #[test]
    fn test_basic_mcmc_ranking() {
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(0, 2, 0.8),
            make_pair(1, 2, 0.7),
        ].into_iter().flatten().collect();

        let opts = default_options();
        let ji = single_judge_info();
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let samples = mcmc.calculate_with_samples(500, 200, 0);
        let ranked = GaussianBT::compute_confidence_intervals_from_sorted_samples(
            &samples.sorted_samples, &samples.means, 0.95,
        );

        assert_eq!(ranked[0].item, 0); // A first
        assert_eq!(ranked[2].item, 2); // C last
    }

    #[test]
    fn test_warm_start() {
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(1, 2, 0.8),
        ].into_iter().flatten().collect();

        let opts = default_options();
        let ji = single_judge_info();
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let _result1 = mcmc.calculate_with_samples(50, 50, 0);
        let state = mcmc.get_current_state();

        let judge_id_to_idx: HashMap<u64, usize> = ji.judge_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let biases = mcmc.get_current_biases(&ji);
        let log_d = mcmc.get_current_log_decisiveness(&ji);

        let mut mcmc2 = GaussianBT::new(3, &results, &opts, &ji);
        let result2 = mcmc2.calculate_incremental_with_samples(&state, &biases, &log_d, &judge_id_to_idx, 50, 0, 0);

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

        let opts = default_options();
        let ji = single_judge_info();
        let mut mcmc = GaussianBT::new(4, &results, &opts, &ji);
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

    #[test]
    fn test_no_logprobs_mode() {
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(0, 2, 0.8),
            make_pair(1, 2, 0.7),
        ].into_iter().flatten().collect();

        let opts = default_options();
        let ji = JudgeInfo {
            judge_ids: vec![0],
            logprobs_mode: false,
        };
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let samples = mcmc.calculate_with_samples(2000, 200, 0);
        let ranked = GaussianBT::compute_confidence_intervals_from_sorted_samples(
            &samples.sorted_samples, &samples.means, 0.95,
        );

        assert_eq!(ranked[0].item, 0);
        assert_eq!(ranked[2].item, 2);
        assert!(mcmc.decisiveness.is_empty());
    }

    #[test]
    fn test_multi_judge_logprobs() {
        // Two judges, judge 0 is more decisive (wider logprob gaps)
        let mut results: Vec<IndexedComparison> = Vec::new();
        // Judge 0: strong opinions
        results.push((0, 1, 0.95, 0));
        results.push((1, 0, 0.05, 0));
        results.push((0, 2, 0.90, 0));
        results.push((2, 0, 0.10, 0));
        // Judge 1: weaker opinions (same direction)
        results.push((0, 1, 0.70, 1));
        results.push((1, 0, 0.30, 1));
        results.push((1, 2, 0.65, 1));
        results.push((2, 1, 0.35, 1));

        let opts = default_options();
        let ji = JudgeInfo {
            judge_ids: vec![100, 200],
            logprobs_mode: true,
        };
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let result = mcmc.calculate_with_samples(500, 200, 0);

        assert_eq!(result.means.len(), 3);
        assert_eq!(result.bias_logit_means.len(), 2);
        assert_eq!(result.log_decisiveness_means.len(), 2);
        assert_eq!(result.comparisons_per_judge.len(), 2);
    }

    #[test]
    fn test_multi_judge_no_logprobs() {
        let mut results: Vec<IndexedComparison> = Vec::new();
        results.push((0, 1, 0.80, 0));
        results.push((1, 0, 0.20, 0));
        results.push((0, 1, 0.75, 1));
        results.push((1, 0, 0.25, 1));
        results.push((1, 2, 0.70, 0));
        results.push((2, 1, 0.30, 0));
        results.push((1, 2, 0.65, 1));
        results.push((2, 1, 0.35, 1));

        let opts = default_options();
        let ji = JudgeInfo {
            judge_ids: vec![100, 200],
            logprobs_mode: false,
        };
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let result = mcmc.calculate_with_samples(500, 200, 0);

        assert_eq!(result.means.len(), 3);
        assert_eq!(result.bias_logit_means.len(), 2);
        assert!(result.log_decisiveness_means.is_empty());
        assert!(result.log_decisiveness_samples.is_empty());
    }

    #[test]
    fn test_single_judge_backward_compat() {
        // Single judge in logprobs mode should force D_1 = 1.0 (ln(D_1) = 0.0)
        let results: Vec<IndexedComparison> = [
            make_pair(0, 1, 0.9),
            make_pair(1, 2, 0.7),
        ].into_iter().flatten().collect();

        let opts = default_options();
        let ji = single_judge_info();
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let result = mcmc.calculate_with_samples(200, 100, 0);

        // With a single judge, geometric mean normalization forces ln(D) = 0 → D = 1.0
        assert_eq!(result.log_decisiveness_means.len(), 1);
        assert!((result.log_decisiveness_means[0]).abs() < 0.01,
            "Single judge decisiveness should be ~0.0 in log space, got {}", result.log_decisiveness_means[0]);
    }
}
