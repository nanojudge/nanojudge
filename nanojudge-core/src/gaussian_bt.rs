/// Ordinal (cumulative-link) Bradley-Terry MCMC sampler with per-judge cutpoints.
///
/// Each comparison is an ordinal verdict on a 5-point scale
/// (A=clear win, B=narrow win, C=draw, D=narrow loss, E=clear loss), supplied by
/// the caller as a per-bucket probability distribution `category_probs`. A judge's
/// full categorical distribution (logprobs mode) or a one-hot vector (text mode) is
/// used directly as a soft observation — nothing is bucketed away. The latent gap is
/// `Δ = θ_i − θ_j`, and each judge owns four free cutpoints `c₁<c₂<c₃<c₄` that
/// slice that axis into the five categories:
///
/// ```text
/// P(E) = σ(c₁ − Δ)
/// P(D) = σ(c₂ − Δ) − σ(c₁ − Δ)
/// P(C) = σ(c₃ − Δ) − σ(c₂ − Δ)
/// P(B) = σ(c₄ − Δ) − σ(c₃ − Δ)
/// P(A) = 1 − σ(c₄ − Δ)
/// ```
///
/// The *center* of a judge's cutpoints absorbs positional bias (no separate bias
/// term); the *spacing* absorbs decisiveness and narrow-vs-clear scale usage. See
/// `benchmark/ORDINAL_MODEL_V2.md`.
///
/// Metropolis-Hastings within Gibbs sampling. Internal module — operates on
/// pre-mapped `usize` indices, not caller IDs.
use std::collections::HashMap;
use rand::Rng;

use crate::types::{IndexedComparison, JudgeInfo, RankedItem, ScoringOptions};

/// Numerically stable `log(sigmoid(x))`.
fn log_sigmoid(x: f64) -> f64 {
    if x > 0.0 {
        -((-x).exp().ln_1p())
    } else {
        x - x.exp().ln_1p()
    }
}

/// Numerically stable `log(1 − exp(−x))` for `x > 0` (Mächler's log1mexp).
fn log1mexp(x: f64) -> f64 {
    if x <= 0.0 {
        f64::NEG_INFINITY
    } else if x < std::f64::consts::LN_2 {
        (-(-x).exp_m1()).ln()
    } else {
        (-(-x).exp()).ln_1p()
    }
}

/// Ordinal verdict on the 5-point scale, from item1's perspective.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    A, // clear win
    B, // narrow win
    C, // draw
    D, // narrow loss
    E, // clear loss
}

/// The five categories in fixed order, indexing `category_probs`.
const VERDICTS: [Verdict; 5] = [Verdict::A, Verdict::B, Verdict::C, Verdict::D, Verdict::E];

/// Soft log-likelihood of a categorical verdict distribution `probs`
/// (`[P(A),P(B),P(C),P(D),P(E)]`) given the latent gap and cutpoints. This is the
/// expected per-category log-probability `Σ_k probs[k]·log P(category k)`, so the
/// judge's full distribution informs the fit with nothing thrown away.
fn dist_loglik(probs: &[f64; 5], delta: f64, c1: f64, c2: f64, c3: f64, c4: f64) -> f64 {
    let mut ll = 0.0;
    for (k, &w) in probs.iter().enumerate() {
        if w > 0.0 {
            ll += w * log_prob_verdict(VERDICTS[k], delta, c1, c2, c3, c4);
        }
    }
    ll
}

/// log P(observed verdict) given the latent gap `delta` and four ordered
/// cutpoints. Differences of sigmoids are expanded via
/// `log(σ(a)−σ(b)) = logσ(a) + logσ(−b) + log1mexp(a−b)` to stay stable.
fn log_prob_verdict(v: Verdict, delta: f64, c1: f64, c2: f64, c3: f64, c4: f64) -> f64 {
    match v {
        Verdict::A => log_sigmoid(delta - c4),
        Verdict::B => log_sigmoid(c4 - delta) + log_sigmoid(delta - c3) + log1mexp(c4 - c3),
        Verdict::C => log_sigmoid(c3 - delta) + log_sigmoid(delta - c2) + log1mexp(c3 - c2),
        Verdict::D => log_sigmoid(c2 - delta) + log_sigmoid(delta - c1) + log1mexp(c2 - c1),
        Verdict::E => log_sigmoid(c1 - delta),
    }
}

// Model constants for the per-judge cutpoint gaps. Cutpoints are parameterized as
// a center `m` plus three positive gaps in log-space:
//   g1 = c2−c1 (narrow-loss band), g2 = c3−c2 (draw width), g3 = c4−c3 (narrow-win band).
// Log-normal priors: g1,g3 median 1, g2 median 2 (weakly informative).
const LG1_PRIOR_MEAN: f64 = 0.0;
const LG2_PRIOR_MEAN: f64 = std::f64::consts::LN_2;
const LG3_PRIOR_MEAN: f64 = 0.0;
const LG_PRIOR_SD: f64 = 1.0;
const GAP_PROPOSAL_STD: f64 = 0.15;

/// Internal representation of a comparison: a categorical verdict distribution
/// from a judge, `[P(A),P(B),P(C),P(D),P(E)]` from item1's perspective.
struct Comparison {
    idx1: usize,
    idx2: usize,
    probs: [f64; 5],
    /// Internal judge index (into the per-judge parameter vecs).
    judge_idx: usize,
}

/// Result from `calculate_with_samples` and `calculate_incremental_with_samples`.
pub struct SamplesResult {
    pub sorted_samples: Vec<Vec<f64>>,
    pub means: Vec<f64>,
    pub top_k_probs: Option<Vec<f64>>,
    /// Per-judge positional-bias samples in logit space. Outer vec indexed by judge.
    pub bias_logit_samples: Vec<Vec<f64>>,
    /// Per-judge positional-bias means in logit space.
    pub bias_logit_means: Vec<f64>,
    /// Unused in the ordinal model (kept for API stability). Always empty.
    pub log_decisiveness_samples: Vec<Vec<f64>>,
    /// Unused in the ordinal model (kept for API stability). Always empty.
    pub log_decisiveness_means: Vec<f64>,
    /// Number of comparisons per judge.
    pub comparisons_per_judge: Vec<usize>,
}

pub struct GaussianBT {
    /// Number of items.
    num_items: usize,
    /// Comparisons (ordinal verdicts).
    comparisons: Vec<Comparison>,
    /// Adjacency list: item_idx -> indices into `comparisons`.
    item_comparisons: Vec<Vec<usize>>,
    /// Current log-strengths (theta values).
    log_strengths: Vec<f64>,
    /// Strength regularization: quadratic shrinkage toward 0 (`-0.5·reg·θ²`).
    regularization_strength: f64,

    /// Number of judges.
    num_judges: usize,
    /// Per-judge cutpoint center (replaces the old positional bias term).
    center: Vec<f64>,
    /// Per-judge log-gaps (c2−c1, c3−c2, c4−c3).
    lg1: Vec<f64>,
    lg2: Vec<f64>,
    lg3: Vec<f64>,
    /// Number of comparisons per judge.
    comparisons_per_judge: Vec<usize>,

    /// Prior on the cutpoint center (logit space).
    center_prior_mu: f64,

    // Hyperparameters (fixed)
    prior_mu: f64,
    prior_tau2: f64,
    proposal_std: f64,
    center_prior_tau2: f64,
    center_proposal_std: f64,
}

impl GaussianBT {
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

        for &(idx1, idx2, probs, judge_idx) in results {
            assert!(idx1 < num_items, "item1 index {} out of range (num_items = {})", idx1, num_items);
            assert!(idx2 < num_items, "item2 index {} out of range (num_items = {})", idx2, num_items);

            let comp_idx = comparisons.len();
            comparisons.push(Comparison {
                idx1,
                idx2,
                probs,
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
            // bias_prior_logit is P(item1 favored) in logit space, but the
            // cutpoint center works in the opposite direction (center > 0
            // penalizes item1) — so the center prior is the NEGATED bias prior.
            center: vec![-options.bias_prior_logit; num_judges],
            lg1: vec![LG1_PRIOR_MEAN; num_judges], // g1 = 1.0
            lg2: vec![LG2_PRIOR_MEAN; num_judges], // g2 = 2.0
            lg3: vec![LG3_PRIOR_MEAN; num_judges], // g3 = 1.0
            comparisons_per_judge,
            center_prior_mu: -options.bias_prior_logit,
            prior_mu,
            prior_tau2: options.prior_tau2,
            proposal_std: options.proposal_std,
            center_prior_tau2: options.bias_prior_tau2,
            center_proposal_std: options.bias_proposal_std,
        }
    }

    /// Build the four ordered cutpoints from a center and three log-gaps. The
    /// cutpoints are placed so their mean equals `center`.
    fn cutpoints_from(center: f64, lg1: f64, lg2: f64, lg3: f64) -> (f64, f64, f64, f64) {
        let g1 = lg1.exp();
        let g2 = lg2.exp();
        let g3 = lg3.exp();
        let c1 = center - (3.0 * g1 + 2.0 * g2 + g3) / 4.0;
        let c2 = c1 + g1;
        let c3 = c2 + g2;
        let c4 = c3 + g3;
        (c1, c2, c3, c4)
    }

    fn cutpoints(&self, j: usize) -> (f64, f64, f64, f64) {
        Self::cutpoints_from(self.center[j], self.lg1[j], self.lg2[j], self.lg3[j])
    }

    fn log_posterior(&self, item_idx: usize, log_strength: f64) -> f64 {
        // Strength prior N(0, prior_tau2) plus quadratic shrinkage regularization.
        let prior_diff = log_strength - self.prior_mu;
        let mut log_prob = -0.5 * prior_diff * prior_diff / self.prior_tau2
            - 0.5 * self.regularization_strength * log_strength * log_strength;

        for &comp_idx in &self.item_comparisons[item_idx] {
            let comp = &self.comparisons[comp_idx];
            let s1 = if comp.idx1 == item_idx { log_strength } else { self.log_strengths[comp.idx1] };
            let s2 = if comp.idx2 == item_idx { log_strength } else { self.log_strengths[comp.idx2] };
            let delta = s1 - s2;
            let (c1, c2, c3, c4) = self.cutpoints(comp.judge_idx);
            log_prob += dist_loglik(&comp.probs, delta, c1, c2, c3, c4);
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

    /// Judge log-likelihood for given cutpoints (built from center + log-gaps).
    fn judge_loglik(&self, judge_idx: usize, center: f64, lg1: f64, lg2: f64, lg3: f64) -> f64 {
        let (c1, c2, c3, c4) = Self::cutpoints_from(center, lg1, lg2, lg3);
        let mut ll = 0.0;
        for comp in &self.comparisons {
            if comp.judge_idx != judge_idx {
                continue;
            }
            let delta = self.log_strengths[comp.idx1] - self.log_strengths[comp.idx2];
            ll += dist_loglik(&comp.probs, delta, c1, c2, c3, c4);
        }
        ll
    }

    fn center_log_prior(&self, center: f64) -> f64 {
        let diff = center - self.center_prior_mu;
        -0.5 * diff * diff / self.center_prior_tau2
    }

    fn gaps_log_prior(lg1: f64, lg2: f64, lg3: f64) -> f64 {
        let a = (lg1 - LG1_PRIOR_MEAN) / LG_PRIOR_SD;
        let b = (lg2 - LG2_PRIOR_MEAN) / LG_PRIOR_SD;
        let c = (lg3 - LG3_PRIOR_MEAN) / LG_PRIOR_SD;
        -0.5 * (a * a + b * b + c * c)
    }

    fn update_center(&mut self, judge_idx: usize, rng: &mut impl Rng) {
        let current = self.center[judge_idx];
        let proposed = current + (rng.random::<f64>() - 0.5) * 2.0 * self.center_proposal_std;
        let (lg1, lg2, lg3) = (self.lg1[judge_idx], self.lg2[judge_idx], self.lg3[judge_idx]);

        let lp_current = self.center_log_prior(current) + self.judge_loglik(judge_idx, current, lg1, lg2, lg3);
        let lp_proposed = self.center_log_prior(proposed) + self.judge_loglik(judge_idx, proposed, lg1, lg2, lg3);

        if rng.random::<f64>().ln() < (lp_proposed - lp_current) {
            self.center[judge_idx] = proposed;
        }
    }

    fn update_gaps(&mut self, judge_idx: usize, rng: &mut impl Rng) {
        let m = self.center[judge_idx];
        let (c1, c2, c3) = (self.lg1[judge_idx], self.lg2[judge_idx], self.lg3[judge_idx]);
        let p1 = c1 + (rng.random::<f64>() - 0.5) * 2.0 * GAP_PROPOSAL_STD;
        let p2 = c2 + (rng.random::<f64>() - 0.5) * 2.0 * GAP_PROPOSAL_STD;
        let p3 = c3 + (rng.random::<f64>() - 0.5) * 2.0 * GAP_PROPOSAL_STD;

        let lp_current = Self::gaps_log_prior(c1, c2, c3) + self.judge_loglik(judge_idx, m, c1, c2, c3);
        let lp_proposed = Self::gaps_log_prior(p1, p2, p3) + self.judge_loglik(judge_idx, m, p1, p2, p3);

        if rng.random::<f64>().ln() < (lp_proposed - lp_current) {
            self.lg1[judge_idx] = p1;
            self.lg2[judge_idx] = p2;
            self.lg3[judge_idx] = p3;
        }
    }

    fn normalize_log_strengths(&mut self) {
        // Center strengths at mean 0 for identifiability.
        let mean = self.log_strengths.iter().sum::<f64>() / self.num_items as f64;
        for val in &mut self.log_strengths {
            *val -= mean;
        }
    }

    fn gibbs_iteration(&mut self, rng: &mut impl Rng) {
        // Step 1: Update each item's log-strength.
        for i in 0..self.num_items {
            self.update_strength(i, rng);
        }

        // Step 2: Update each judge's cutpoints (center, then the three gaps).
        for k in 0..self.num_judges {
            self.update_center(k, rng);
            self.update_gaps(k, rng);
        }
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

        for _ in 0..iterations {
            self.gibbs_iteration(rng);
            self.normalize_log_strengths();

            for idx in 0..n {
                samples_per_item[idx].push(self.log_strengths[idx]);
            }

            // Positional bias = negated center: center>0 leans toward item2, so
            // -center as a logit gives P(item1 wins | equal strength) > 0.5 when
            // the judge favors the first item.
            for j in 0..k {
                bias_samples[j].push(-self.center[j]);
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

        let mut bias_logit_means = Vec::with_capacity(k);
        for j in 0..k {
            bias_logit_means.push(bias_samples[j].iter().sum::<f64>() / bias_samples[j].len() as f64);
            bias_samples[j].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        SamplesResult {
            sorted_samples,
            means,
            top_k_probs: top_k_count.map(|c| c.iter().map(|&v| v as f64 / iterations as f64).collect()),
            bias_logit_samples: bias_samples,
            bias_logit_means,
            log_decisiveness_samples: Vec::new(),
            log_decisiveness_means: Vec::new(),
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

    /// Get current per-judge cutpoint centers (keyed by judge_id from the info).
    pub fn get_current_biases(&self, judge_info: &JudgeInfo) -> Vec<(u64, f64)> {
        judge_info.judge_ids.iter().enumerate()
            .map(|(idx, &id)| (id, self.center[idx]))
            .collect()
    }

    /// Unused in the ordinal model (kept for API stability). Always empty.
    pub fn get_current_log_decisiveness(&self, _judge_info: &JudgeInfo) -> Vec<(u64, f64)> {
        Vec::new()
    }

    /// Warm-start MCMC returning raw sorted samples.
    pub fn calculate_incremental_with_samples(
        &mut self,
        previous_strengths: &[f64],
        previous_biases: &[(u64, f64)],
        _previous_log_decisiveness: &[(u64, f64)],
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

        // Restore per-judge cutpoint centers (gaps re-burn from defaults).
        for &(judge_id, center) in previous_biases {
            if let Some(&idx) = judge_id_to_idx.get(&judge_id) {
                self.center[idx] = center;
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
            logprobs_mode: false,
        }
    }

    /// One-hot category distribution for a scalar win probability, mirroring the
    /// five ordinal buckets. Lets the scalar-based tests express discrete verdicts.
    fn dist(p: f64) -> [f64; 5] {
        if p > 0.9 { [1.0, 0.0, 0.0, 0.0, 0.0] }
        else if p > 0.65 { [0.0, 1.0, 0.0, 0.0, 0.0] }
        else if p > 0.35 { [0.0, 0.0, 1.0, 0.0, 0.0] }
        else if p > 0.1 { [0.0, 0.0, 0.0, 1.0, 0.0] }
        else { [0.0, 0.0, 0.0, 0.0, 1.0] }
    }

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(i1: usize, i2: usize, prob: f64) -> [IndexedComparison; 2] {
        [(i1, i2, dist(prob), 0), (i2, i1, dist(1.0 - prob), 0)]
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
        let samples = mcmc.calculate_with_samples(20000, 500, 0);
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
    fn test_multi_judge() {
        let mut results: Vec<IndexedComparison> = Vec::new();
        results.push((0, 1, dist(0.80), 0));
        results.push((1, 0, dist(0.20), 0));
        results.push((0, 1, dist(0.75), 1));
        results.push((1, 0, dist(0.25), 1));
        results.push((1, 2, dist(0.70), 0));
        results.push((2, 1, dist(0.30), 0));
        results.push((1, 2, dist(0.65), 1));
        results.push((2, 1, dist(0.35), 1));

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
        assert_eq!(result.comparisons_per_judge.len(), 2);
    }
}
