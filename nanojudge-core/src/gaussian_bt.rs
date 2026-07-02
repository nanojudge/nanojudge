/// Linear Bradley-Terry MCMC sampler with per-judge positional bias.
///
/// Each comparison carries a scalar win probability `p` (item1's perspective),
/// collapsed from the 4-category scale by the caller. The likelihood is:
///
/// ```text
/// LL = p·log σ(d) + (1−p)·log σ(−d)
/// where d = θ_i − θ_j + γ_k
/// ```
///
/// `θ_i`, `θ_j` are item log-strengths, `γ_k` is per-judge positional bias
/// (positive = favors first-listed item).
///
/// Metropolis-Hastings within Gibbs sampling. Internal module — operates on
/// pre-mapped `usize` indices, not caller IDs.
use std::collections::HashMap;
use rand::Rng;
use rand::rngs::StdRng;

use crate::types::{IndexedComparison, JudgeInfo, RankedItem, ScoringOptions};

/// Numerically stable `log(sigmoid(x))`.
fn log_sigmoid(x: f64) -> f64 {
    if x > 0.0 {
        -((-x).exp().ln_1p())
    } else {
        x - x.exp().ln_1p()
    }
}

/// Log-likelihood of a single comparison with win probability `p` and latent
/// gap `d = θ_i − θ_j + γ_k`.
fn comparison_loglik(p: f64, d: f64) -> f64 {
    p * log_sigmoid(d) + (1.0 - p) * log_sigmoid(-d)
}

/// Internal representation of a comparison.
struct Comparison {
    idx1: usize,
    idx2: usize,
    /// P(item1 wins), collapsed from the 4-category distribution.
    win_prob: f64,
    /// Internal judge index (into the per-judge parameter vecs).
    judge_idx: usize,
}

/// Result from `calculate_with_samples` and `calculate_incremental_with_samples`.
pub struct SamplesResult {
    pub sorted_samples: Vec<Vec<f64>>,
    pub means: Vec<f64>,
    /// Per-item posterior standard deviation (spread of the strength samples),
    /// in the same order as `means`. Used to summarize each item as a Gaussian
    /// `(mean, std)` for top-heavy selection weighting.
    pub stds: Vec<f64>,
    /// Per-judge positional-bias samples in logit space. Outer vec indexed by judge.
    pub bias_logit_samples: Vec<Vec<f64>>,
    /// Per-judge positional-bias means in logit space.
    pub bias_logit_means: Vec<f64>,
    /// Panel-level positional-bias samples in probability space, sorted ascending.
    /// Each sample is the comparison-count-weighted average of the judges' bias
    /// probabilities at one MCMC iteration (equal weights when there are no
    /// comparisons).
    pub panel_bias_samples: Vec<f64>,
    /// Number of comparisons per judge.
    pub comparisons_per_judge: Vec<usize>,
}

pub struct GaussianBT {
    /// Number of items.
    num_items: usize,
    /// Comparisons.
    comparisons: Vec<Comparison>,
    /// Adjacency list: item_idx -> indices into `comparisons`.
    item_comparisons: Vec<Vec<usize>>,
    /// Current log-strengths (theta values).
    log_strengths: Vec<f64>,
    /// Strength regularization: quadratic shrinkage toward 0 (`-0.5·reg·θ²`).
    regularization_strength: f64,

    /// Number of judges.
    num_judges: usize,
    /// Per-judge positional bias (logit space, positive = favors item1).
    bias: Vec<f64>,
    /// Number of comparisons per judge.
    comparisons_per_judge: Vec<usize>,

    // Hyperparameters (fixed)
    prior_mu: f64,
    prior_tau2: f64,
    proposal_std: f64,
    bias_prior_mu: f64,
    bias_prior_tau2: f64,
    bias_proposal_std: f64,
}

impl GaussianBT {
    /// Build the sampler from pre-indexed comparisons.
    ///
    /// # Panics
    ///
    /// - if any comparison's item index is `>= num_items`
    /// - if any comparison's judge index is `>= judge_info.judge_ids.len()`
    pub fn new(
        num_items: usize,
        results: &[IndexedComparison],
        options: &ScoringOptions,
        judge_info: &JudgeInfo,
    ) -> Self {
        let prior_mu = 0.0;
        let num_judges = judge_info.judge_ids.len();

        let mut comparisons = Vec::new();
        let mut item_comparisons: Vec<Vec<usize>> = (0..num_items).map(|_| Vec::new()).collect();
        let mut comparisons_per_judge = vec![0usize; num_judges];

        for &(idx1, idx2, win_prob, judge_idx) in results {
            assert!(idx1 < num_items, "item1 index {} out of range (num_items = {})", idx1, num_items);
            assert!(idx2 < num_items, "item2 index {} out of range (num_items = {})", idx2, num_items);

            let comp_idx = comparisons.len();
            comparisons.push(Comparison {
                idx1,
                idx2,
                win_prob,
                judge_idx,
            });
            item_comparisons[idx1].push(comp_idx);
            item_comparisons[idx2].push(comp_idx);
            comparisons_per_judge[judge_idx] += 1;
        }

        GaussianBT {
            num_items,
            comparisons,
            item_comparisons,
            log_strengths: vec![prior_mu; num_items],
            regularization_strength: options.regularization_strength,
            num_judges,
            bias: vec![options.bias_prior_logit; num_judges],
            comparisons_per_judge,
            prior_mu,
            prior_tau2: options.prior_tau2,
            proposal_std: options.proposal_std,
            bias_prior_mu: options.bias_prior_logit,
            bias_prior_tau2: options.bias_prior_tau2,
            bias_proposal_std: options.bias_proposal_std,
        }
    }

    fn log_posterior(&self, item_idx: usize, log_strength: f64) -> f64 {
        let prior_diff = log_strength - self.prior_mu;
        let mut log_prob = -0.5 * prior_diff * prior_diff / self.prior_tau2
            - 0.5 * self.regularization_strength * log_strength * log_strength;

        for &comp_idx in &self.item_comparisons[item_idx] {
            let comp = &self.comparisons[comp_idx];
            let s1 = if comp.idx1 == item_idx { log_strength } else { self.log_strengths[comp.idx1] };
            let s2 = if comp.idx2 == item_idx { log_strength } else { self.log_strengths[comp.idx2] };
            let d = s1 - s2 + self.bias[comp.judge_idx];
            log_prob += comparison_loglik(comp.win_prob, d);
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

    /// Judge log-likelihood for a given bias value.
    fn judge_loglik(&self, judge_idx: usize, bias: f64) -> f64 {
        let mut ll = 0.0;
        for comp in &self.comparisons {
            if comp.judge_idx != judge_idx {
                continue;
            }
            let d = self.log_strengths[comp.idx1] - self.log_strengths[comp.idx2] + bias;
            ll += comparison_loglik(comp.win_prob, d);
        }
        ll
    }

    fn bias_log_prior(&self, bias: f64) -> f64 {
        let diff = bias - self.bias_prior_mu;
        -0.5 * diff * diff / self.bias_prior_tau2
    }

    fn update_bias(&mut self, judge_idx: usize, rng: &mut impl Rng) {
        let current = self.bias[judge_idx];
        let proposed = current + (rng.random::<f64>() - 0.5) * 2.0 * self.bias_proposal_std;

        let lp_current = self.bias_log_prior(current) + self.judge_loglik(judge_idx, current);
        let lp_proposed = self.bias_log_prior(proposed) + self.judge_loglik(judge_idx, proposed);

        if rng.random::<f64>().ln() < (lp_proposed - lp_current) {
            self.bias[judge_idx] = proposed;
        }
    }

    fn normalize_log_strengths(&mut self) {
        let mean = self.log_strengths.iter().sum::<f64>() / self.num_items as f64;
        for val in &mut self.log_strengths {
            *val -= mean;
        }
    }

    fn gibbs_iteration(&mut self, rng: &mut impl Rng) {
        for i in 0..self.num_items {
            self.update_strength(i, rng);
        }

        for k in 0..self.num_judges {
            self.update_bias(k, rng);
        }
    }

    /// Run MCMC sampling loop and collect results.
    fn collect_samples(
        &mut self,
        iterations: usize,
        rng: &mut impl Rng,
    ) -> SamplesResult {
        let n = self.num_items;
        let k = self.num_judges;

        let mut samples_per_item: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(iterations)).collect();
        let mut bias_samples: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(iterations)).collect();

        let total_judge_comparisons: usize = self.comparisons_per_judge.iter().sum();
        let panel_weights: Vec<f64> = if total_judge_comparisons > 0 {
            self.comparisons_per_judge.iter().map(|&c| c as f64 / total_judge_comparisons as f64).collect()
        } else {
            vec![1.0 / k as f64; k]
        };
        let mut panel_bias_samples: Vec<f64> = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            self.gibbs_iteration(rng);
            self.normalize_log_strengths();

            for (idx, samples) in samples_per_item.iter_mut().enumerate().take(n) {
                samples.push(self.log_strengths[idx]);
            }

            let mut panel_prob = 0.0;
            for j in 0..k {
                let bias_logit = self.bias[j];
                bias_samples[j].push(bias_logit);
                panel_prob += panel_weights[j] / (1.0 + (-bias_logit).exp());
            }
            panel_bias_samples.push(panel_prob);
        }

        let mut sorted_samples = Vec::with_capacity(n);
        let mut means = Vec::with_capacity(n);
        let mut stds = Vec::with_capacity(n);

        for samples in samples_per_item.iter_mut().take(n) {
            let count = samples.len() as f64;
            let mean = samples.iter().sum::<f64>() / count;
            // Population variance of the collected samples → posterior spread.
            let variance = samples.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / count;
            means.push(mean);
            stds.push(variance.max(0.0).sqrt());
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted_samples.push(std::mem::take(samples));
        }

        let mut bias_logit_means = Vec::with_capacity(k);
        for bias_sample in bias_samples.iter_mut().take(k) {
            bias_logit_means.push(bias_sample.iter().sum::<f64>() / bias_sample.len() as f64);
            bias_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        panel_bias_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        SamplesResult {
            sorted_samples,
            means,
            stds,
            bias_logit_samples: bias_samples,
            bias_logit_means,
            panel_bias_samples,
            comparisons_per_judge: self.comparisons_per_judge.clone(),
        }
    }

    /// Cold-start MCMC returning raw sorted samples.
    pub fn calculate_with_samples(
        &mut self,
        mcmc_iterations: usize,
        burn_in: usize,
        rng: &mut StdRng,
    ) -> SamplesResult {
        for _ in 0..burn_in {
            self.gibbs_iteration(rng);
            self.normalize_log_strengths();
        }

        self.collect_samples(mcmc_iterations, rng)
    }

    /// Get current item state for warm-starting (log-strengths for real items).
    pub fn get_current_state(&self) -> Vec<f64> {
        self.log_strengths[..self.num_items].to_vec()
    }

    /// Get current per-judge biases (keyed by judge_id from the info).
    pub fn get_current_biases(&self, judge_info: &JudgeInfo) -> Vec<(u64, f64)> {
        judge_info.judge_ids.iter().enumerate()
            .map(|(idx, &id)| (id, self.bias[idx]))
            .collect()
    }

    /// Warm-start MCMC returning raw sorted samples.
    ///
    /// # Panics
    ///
    /// Panics if `previous_log_strengths` does not have one entry per item.
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_incremental_with_samples(
        &mut self,
        previous_log_strengths: &[f64],
        previous_biases: &[(u64, f64)],
        judge_id_to_idx: &HashMap<u64, usize>,
        new_iterations: usize,
        burn_in: usize,
        rng: &mut StdRng,
    ) -> SamplesResult {
        let n = self.num_items;
        assert_eq!(previous_log_strengths.len(), n, "Previous state size mismatch");

        self.log_strengths[..n].copy_from_slice(&previous_log_strengths[..n]);

        for &(judge_id, bias) in previous_biases {
            if let Some(&idx) = judge_id_to_idx.get(&judge_id) {
                self.bias[idx] = bias;
            }
        }

        for _ in 0..burn_in {
            self.gibbs_iteration(rng);
            self.normalize_log_strengths();
        }

        self.collect_samples(new_iterations, rng)
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
    use crate::seed;
    use crate::types::{InferenceMode, ScoringOptions};

    fn single_judge_info() -> JudgeInfo {
        JudgeInfo {
            judge_ids: vec![0],
            logprobs_mode: false,
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
            selection_sharpness: None,
            selection_cutoff: 0.05,
            selection_coverage: 0.0,
            target_prior_games: 10.0,
            warm_start: None,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            proposal_std: 0.3,
            bias_prior_tau2: 2.0,
            bias_proposal_std: 0.15,
            bias_prior_logit: 0.0,
            seed: None,
            inference: InferenceMode::Mcmc,
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
        let mut rng = seed::make_rng(None, seed::SUBSYSTEM_MCMC);
        let samples = mcmc.calculate_with_samples(20000, 500, &mut rng);
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
        let mut rng = seed::make_rng(None, seed::SUBSYSTEM_MCMC);
        let _result1 = mcmc.calculate_with_samples(50, 50, &mut rng);
        let state = mcmc.get_current_state();

        let judge_id_to_idx: HashMap<u64, usize> = ji.judge_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let biases = mcmc.get_current_biases(&ji);

        let mut mcmc2 = GaussianBT::new(3, &results, &opts, &ji);
        let mut rng2 = seed::make_rng(None, seed::SUBSYSTEM_MCMC);
        let result2 = mcmc2.calculate_incremental_with_samples(&state, &biases, &judge_id_to_idx, 50, 0, &mut rng2);

        assert_eq!(result2.means.len(), 3);
    }

    #[test]
    fn test_stds_reported_per_item() {
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
        let mut rng = seed::make_rng(None, seed::SUBSYSTEM_MCMC);
        let result = mcmc.calculate_with_samples(200, 100, &mut rng);

        // One posterior std per item, all finite and non-negative.
        assert_eq!(result.stds.len(), 4);
        assert_eq!(result.means.len(), 4);
        for s in &result.stds {
            assert!(s.is_finite() && *s >= 0.0, "std must be finite and non-negative, got {s}");
        }
        // With real comparison data the chain moves, so spread is strictly positive.
        assert!(result.stds.iter().all(|&s| s > 0.0), "expected positive spread with data");
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
    fn test_multi_judge() {
        let results: Vec<IndexedComparison> = vec![
            (0, 1, 0.80, 0),
            (1, 0, 0.20, 0),
            (0, 1, 0.75, 1),
            (1, 0, 0.25, 1),
            (1, 2, 0.70, 0),
            (2, 1, 0.30, 0),
            (1, 2, 0.65, 1),
            (2, 1, 0.35, 1),
        ];

        let opts = default_options();
        let ji = JudgeInfo {
            judge_ids: vec![100, 200],
            logprobs_mode: false,
        };
        let mut mcmc = GaussianBT::new(3, &results, &opts, &ji);
        let mut rng = seed::make_rng(None, seed::SUBSYSTEM_MCMC);
        let result = mcmc.calculate_with_samples(500, 200, &mut rng);

        assert_eq!(result.means.len(), 3);
        assert_eq!(result.bias_logit_means.len(), 2);
        assert_eq!(result.comparisons_per_judge.len(), 2);
    }
}
