/// Unified MCMC scoring wrapper.
///
/// One function, one options struct. Pure function — no IO, no state.
/// Items are identified by caller-provided `i64` IDs.
use std::collections::HashMap;

use crate::gaussian_bt::GaussianBT;
use crate::laplace_bt;
use crate::seed;
use crate::types::{
    ComparisonInput, IdMap, IndexedComparison, InferenceMode, JudgeAnalytics, JudgeInfo,
    RankedItem, ScoringOptions, ScoringResult, WarmStartState,
};

/// Compute confidence interval from sorted samples.
fn ci_from_sorted(samples: &[f64], confidence_level: f64) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let alpha = 1.0 - confidence_level;
    let n = samples.len();
    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = upper_idx.saturating_sub(1).max(lower_idx);
    (samples[lower_idx], samples[upper_idx])
}

/// Convert logit-space samples to probability-space CI.
fn logit_to_prob_ci(sorted_logit_samples: &[f64], mean_logit: f64, confidence_level: f64) -> (f64, (f64, f64)) {
    let prob = 1.0 / (1.0 + (-mean_logit).exp());
    let (lower_logit, upper_logit) = ci_from_sorted(sorted_logit_samples, confidence_level);
    let lower = 1.0 / (1.0 + (-lower_logit).exp());
    let upper = 1.0 / (1.0 + (-upper_logit).exp());
    (prob, (lower, upper))
}

/// Standard normal CDF Φ(x).
///
/// Uses the Abramowitz & Stegun 7.1.26 rational approximation of `erf`
/// (|absolute error| < 1.5e-7) via `Φ(x) = 0.5·(1 + erf(x/√2))`. That precision
/// is far beyond what a sampling weight needs, and it keeps the crate
/// dependency-free.
fn normal_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = (x / std::f64::consts::SQRT_2).abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    0.5 * (1.0 + sign * erf)
}

/// Build top-heavy item-selection weights from each item's posterior summary.
///
/// For each item, split its posterior at the anchor's mean: with
/// `A = P(strength_i ≥ anchor_mean)` under a `Gaussian(mean, std)` summary,
/// the base weight is the uncertainty ratio `min(A, 1−A) / max(A, 1−A)` —
/// 1 when the posterior sits astride the anchor (maximally unsure which side
/// the item belongs on), decaying toward 0 as the item resolves confidently
/// above or below it — raised to `selection_sharpness` (lower = flatter =
/// more exploration). The anchor is the item at rank `anchor_index` (0-based,
/// posterior means sorted descending; fractional values interpolate between
/// the two adjacent ranks); the anchor itself sits at ratio 1. Items whose
/// ratio is below `selection_cutoff` are dropped to 0, except the two
/// highest-ratio items, which are always kept so the pairing layer can always
/// draw two distinct contenders.
///
/// The base weight is then divided by `games_played^selection_coverage` — a
/// proportional-fair coverage pull that drives each item's cumulative comparison
/// count toward its ratio-implied share (`selection_coverage = 0` disables it, `1`
/// is standard proportional-fair). `games_played` uses the current cumulative
/// counts, so a resolved item (ratio → 0) sheds its weight rather than carrying
/// stale "owed" comparisons.
///
/// Weights are returned unnormalized; the pairing layer normalizes by its
/// running total.
#[allow(clippy::too_many_arguments)]
fn compute_selection_weights(
    means: &[f64],
    stds: &[f64],
    games_played: &[usize],
    selection_sharpness: f64,
    anchor_index: f64,
    selection_cutoff: f64,
    selection_coverage: f64,
    prior_tau2: f64,
    target_prior_games: f64,
) -> Vec<f64> {
    let n = means.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(
        anchor_index.is_finite() && anchor_index >= 0.0 && anchor_index <= (n - 1) as f64,
        "anchor_index={anchor_index}, must be finite and in [0, num_items - 1 = {}]",
        n - 1
    );

    // The anchor: the item at rank `anchor_index` when posterior means are
    // sorted descending. Fractional indices interpolate linearly between the
    // two adjacent ranks — for both the anchor's mean and its game count.
    let mut by_mean: Vec<usize> = (0..n).collect();
    by_mean.sort_by(|&a, &b| means[b].partial_cmp(&means[a]).unwrap_or(std::cmp::Ordering::Equal));
    let lo_rank = anchor_index.floor() as usize;
    let hi_rank = anchor_index.ceil() as usize;
    let frac = anchor_index - lo_rank as f64;
    let (lo_idx, hi_idx) = (by_mean[lo_rank], by_mean[hi_rank]);
    let observed_anchor = means[lo_idx] + frac * (means[hi_idx] - means[lo_idx]);

    // The selection target: the reference strength each item's uncertainty
    // ratio is measured against. The observed anchor is unreliable early — it has few games and,
    // in binary mode, wins never pin its magnitude — so blend it with a
    // prior-predicted anchor via a pseudo-count. The prediction is worth
    // `target_prior_games` games against the anchor's actual game count `g`:
    //   `target = (g·observed_anchor + K·predicted_anchor) / (g + K)`.
    // `target_prior_games = 0` falls straight back to the observed anchor.
    let target = if target_prior_games > 0.0 {
        let predicted_anchor = predicted_rank_strength(means, prior_tau2, anchor_index);
        let g_lo = games_played[lo_idx] as f64;
        let g_hi = games_played[hi_idx] as f64;
        let g = g_lo + frac * (g_hi - g_lo);
        (g * observed_anchor + target_prior_games * predicted_anchor) / (g + target_prior_games)
    } else {
        observed_anchor
    };

    // Uncertainty ratio per item: split the posterior at the target into the
    // area above (A) and below (1−A); the ratio min/max is 1 when the item
    // straddles the target and decays toward 0 once it is confidently on
    // either side. max(A, 1−A) >= 0.5, so the division is always safe.
    let ratios: Vec<f64> = (0..n)
        .map(|i| {
            let above = if stds[i] <= 1e-12 {
                // Degenerate point mass: strictly-above the target counts fully,
                // exactly-at counts half, below counts nothing.
                if means[i] > target {
                    1.0
                } else if means[i] >= target {
                    0.5
                } else {
                    0.0
                }
            } else {
                normal_cdf((means[i] - target) / stds[i])
            };
            let below = 1.0 - above;
            above.min(below) / above.max(below)
        })
        .collect();

    // Always keep the two highest-ratio items regardless of cutoff, so the
    // candidate pool never collapses below the two needed to form a pair.
    let mut kept = vec![false; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ratios[b].partial_cmp(&ratios[a]).unwrap_or(std::cmp::Ordering::Equal));
    for &idx in order.iter().take(2) {
        kept[idx] = true;
    }

    (0..n)
        .map(|i| {
            let base = if ratios[i] >= selection_cutoff || kept[i] {
                ratios[i].powf(selection_sharpness)
            } else {
                0.0
            };
            // Proportional-fair coverage: divide by games-played^selection_coverage. Guard
            // the denominator at 1 so an item not yet played doesn't blow up
            // (the uniform stage guarantees >= min_uniform_games before these
            // weights are actually used for pairing).
            let served = (games_played[i] as f64).max(1.0);
            base / served.powf(selection_coverage)
        })
        .collect()
}

/// Run MCMC scoring on pairwise comparison data.
///
/// `item_ids` is the full list of item IDs being ranked. The returned
/// `selection_weights` (when requested) is in the same order as `item_ids`.
///
/// # Panics
///
/// Panics if caller-supplied data violates the input contract:
/// - `judge_info.judge_ids` is empty (a tournament needs at least one judge)
/// - `item_ids` contains a duplicate ID
/// - a comparison references an item ID not present in `item_ids`
/// - a comparison's `judge_id` is not present in `judge_info.judge_ids`
/// - `options.warm_start` is set and its `item_log_strengths` length does not
///   match `item_ids.len()`
/// - `options.selection_sharpness` is set and `options.anchor_index` is not
///   finite or lies outside `[0, item_ids.len() - 1]`
pub fn run_scoring(
    item_ids: &[i64],
    comparisons: &[ComparisonInput],
    options: &ScoringOptions,
    judge_info: &JudgeInfo,
) -> ScoringResult {
    assert!(options.iterations > 0, "iterations must be at least 1");
    assert!(
        !judge_info.judge_ids.is_empty(),
        "judge_info.judge_ids must contain at least one judge"
    );

    let id_map = IdMap::from_ids(item_ids);
    let num_items = id_map.len();

    // Build judge_id -> internal index mapping
    let mut judge_id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(judge_info.judge_ids.len());
    for (idx, &id) in judge_info.judge_ids.iter().enumerate() {
        judge_id_to_idx.insert(id, idx);
    }

    let indexed = id_map.convert_comparisons(comparisons, &judge_id_to_idx);

    // Laplace path: deterministic MAP + curvature, no sampling. Produces the
    // same ScoringResult shape as the MCMC path below.
    if options.inference == InferenceMode::LaplaceLinear {
        return run_scoring_laplace(&id_map, num_items, &indexed, options, judge_info);
    }

    let mut mcmc = GaussianBT::new(
        num_items,
        &indexed,
        options,
        judge_info,
    );

    let mut rng = seed::make_rng(options.seed, seed::SUBSYSTEM_MCMC);

    let samples_result = if let Some(ref warm_start) = options.warm_start {
        assert_eq!(
            warm_start.item_log_strengths.len(), num_items,
            "warm_start item_log_strengths length ({}) must match num_items ({})",
            warm_start.item_log_strengths.len(), num_items
        );
        mcmc.calculate_incremental_with_samples(
            &warm_start.item_log_strengths,
            &warm_start.judge_biases,
            &judge_id_to_idx,
            options.iterations,
            options.burn_in,
            &mut rng,
        )
    } else {
        mcmc.calculate_with_samples(options.iterations, options.burn_in, &mut rng)
    };

    // Compute confidence intervals; returned items use index-as-i64, map back to real IDs
    let mut rankings = GaussianBT::compute_confidence_intervals_from_sorted_samples(
        &samples_result.sorted_samples,
        &samples_result.means,
        options.confidence_level,
    );

    for r in &mut rankings {
        r.item = id_map.to_id(r.item as usize);
    }

    // Build per-judge analytics
    let mut judge_analytics = Vec::with_capacity(judge_info.judge_ids.len());
    for (j, &judge_id) in judge_info.judge_ids.iter().enumerate() {
        let (bias_prob, bias_ci) = logit_to_prob_ci(
            &samples_result.bias_logit_samples[j],
            samples_result.bias_logit_means[j],
            options.confidence_level,
        );

        judge_analytics.push(JudgeAnalytics {
            judge_id,
            positional_bias: bias_prob,
            positional_bias_ci: bias_ci,
            num_comparisons: samples_result.comparisons_per_judge[j],
        });
    }

    // Panel-level bias: posterior mean and quantiles of the per-iteration
    // weighted average of judge biases (probability space).
    let panel_samples = &samples_result.panel_bias_samples;
    let panel_positional_bias =
        panel_samples.iter().sum::<f64>() / panel_samples.len() as f64;
    let panel_positional_bias_ci = ci_from_sorted(panel_samples, options.confidence_level);

    // Build warm start state
    let warm_start_state = WarmStartState {
        item_log_strengths: mcmc.get_current_state(),
        judge_biases: mcmc.get_current_biases(judge_info),
    };

    // Top-heavy selection weights: each item's sharpened uncertainty ratio
    // around the anchor's mean, built from the posterior (mean, std) summary,
    // with a proportional-fair coverage pull. `None` for uniform.
    let selection_weights = options.selection_sharpness.map(|selection_sharpness| {
        // Comparisons played per item so far (each comparison touches two items).
        let mut games_per_item = vec![0usize; num_items];
        for &(idx1, idx2, _, _, _, _) in &indexed {
            games_per_item[idx1] += 1;
            games_per_item[idx2] += 1;
        }
        compute_selection_weights(
            &samples_result.means,
            &samples_result.stds,
            &games_per_item,
            selection_sharpness,
            options.anchor_index,
            options.selection_cutoff,
            options.selection_coverage,
            options.prior_tau2,
            options.target_prior_games,
        )
    });

    ScoringResult {
        rankings,
        selection_weights,
        warm_start_state,
        sample_size: options.iterations,
        judge_analytics,
        panel_positional_bias,
        panel_positional_bias_ci,
    }
}

/// Inverse standard-normal CDF for a confidence level, via bisection on
/// `normal_cdf` (run once per scoring call). Returns `z` such that
/// `normal_cdf(z) = 1 − (1 − confidence_level)/2`.
fn z_for_confidence(confidence_level: f64) -> f64 {
    let target = 1.0 - (1.0 - confidence_level) / 2.0;
    let (mut lo, mut hi) = (0.0_f64, 10.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if normal_cdf(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Inverse standard-normal CDF (probit) for an arbitrary probability `p` in
/// (0, 1), via bisection on `normal_cdf`. Returns `z` such that
/// `normal_cdf(z) = p`. Used to place the prior-predicted anchor strength.
fn inverse_normal_cdf(p: f64) -> f64 {
    let (mut lo, mut hi) = (-10.0_f64, 10.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Prior-predicted strength of the item at rank `anchor_index` (0-based from
/// the top, fractional allowed): the expected `(anchor_index + 1)`-th largest
/// of `n` draws from the strength prior `N(center, prior_tau2)`, where `center`
/// is the observed mean of the posterior means (which is well-determined even
/// when individual magnitudes are not). The expected order statistic is
/// approximated by Blom's plotting position for the r-th largest,
/// `Φ⁻¹((n − r + 0.625)/(n + 0.25))` with `r = anchor_index + 1`, scaled by the
/// prior std and shifted to the observed center. Continuous in `anchor_index`,
/// so fractional anchors need no separate interpolation; `anchor_index = 0`
/// reproduces the classic `E[max]` position `Φ⁻¹((n − 0.375)/(n + 0.25))`.
fn predicted_rank_strength(means: &[f64], prior_tau2: f64, anchor_index: f64) -> f64 {
    let n = means.len();
    let center = means.iter().sum::<f64>() / n as f64;
    let sigma0 = prior_tau2.max(0.0).sqrt();
    let nf = n as f64;
    let p = (nf - anchor_index - 0.375) / (nf + 0.25);
    center + sigma0 * inverse_normal_cdf(p)
}

/// Numerically stable sigmoid.
fn sigmoid_scalar(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Laplace-approximation scoring path: deterministic MAP fit + curvature, no
/// sampling. Produces the same `ScoringResult` shape as the MCMC path, with CIs
/// computed analytically as symmetric Gaussian intervals from the per-parameter
/// standard deviations.
#[allow(clippy::needless_range_loop)]
fn run_scoring_laplace(
    id_map: &IdMap,
    num_items: usize,
    indexed: &[IndexedComparison],
    options: &ScoringOptions,
    judge_info: &JudgeInfo,
) -> ScoringResult {
    const MAX_NEWTON_ITERS: usize = 100;
    const NEWTON_TOL: f64 = 1e-8;

    let num_judges = judge_info.judge_ids.len();
    let fit = laplace_bt::fit_linear(
        num_items,
        num_judges,
        indexed,
        options.prior_tau2,
        options.regularization_strength,
        options.bias_prior_logit,
        options.bias_prior_tau2,
        MAX_NEWTON_ITERS,
        NEWTON_TOL,
    );

    let z = z_for_confidence(options.confidence_level);

    // Rankings: score = MAP log-strength, symmetric Gaussian CI = mean ± z·std.
    let mut rankings: Vec<RankedItem> = (0..num_items)
        .map(|i| RankedItem {
            item: id_map.to_id(i),
            score: fit.means[i],
            lower_bound: fit.means[i] - z * fit.stds[i],
            upper_bound: fit.means[i] + z * fit.stds[i],
        })
        .collect();
    rankings.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Per-item and per-judge comparison counts.
    let mut games_per_item = vec![0usize; num_items];
    let mut comparisons_per_judge = vec![0usize; num_judges];
    for &(i, j, _, k, _, _) in indexed {
        games_per_item[i] += 1;
        games_per_item[j] += 1;
        comparisons_per_judge[k] += 1;
    }

    // Selection weights reuse the shared Gaussian-area helper on (mean, std).
    let selection_weights = options.selection_sharpness.map(|sharpness| {
        compute_selection_weights(
            &fit.means,
            &fit.stds,
            &games_per_item,
            sharpness,
            options.anchor_index,
            options.selection_cutoff,
            options.selection_coverage,
            options.prior_tau2,
            options.target_prior_games,
        )
    });

    // The Laplace fit refits from scratch each call, so warm start is unused;
    // we still return a valid state for API symmetry with the MCMC path.
    let warm_start_state = WarmStartState {
        item_log_strengths: fit.means.clone(),
        judge_biases: (0..num_judges)
            .map(|k| (judge_info.judge_ids[k], fit.bias_free[k].clone()))
            .collect(),
    };

    // Per-judge analytics: bias in probability space with a symmetric CI.
    let judge_analytics: Vec<JudgeAnalytics> = (0..num_judges)
        .map(|k| {
            let b = fit.bias_means[k];
            let s = fit.bias_stds[k];
            JudgeAnalytics {
                judge_id: judge_info.judge_ids[k],
                positional_bias: sigmoid_scalar(b),
                positional_bias_ci: (sigmoid_scalar(b - z * s), sigmoid_scalar(b + z * s)),
                num_comparisons: comparisons_per_judge[k],
            }
        })
        .collect();

    // Panel bias: comparison-count-weighted average of the judges' bias
    // probabilities (equal weights when there are no comparisons), with the CI
    // propagated from the per-judge bias variances via the delta method.
    let total_comparisons: usize = comparisons_per_judge.iter().sum();
    let (panel_positional_bias, panel_var) = if num_judges == 0 {
        (0.5, 0.0)
    } else {
        let mut mean = 0.0;
        let mut var = 0.0;
        for k in 0..num_judges {
            let w = if total_comparisons == 0 {
                1.0 / num_judges as f64
            } else {
                comparisons_per_judge[k] as f64 / total_comparisons as f64
            };
            let p = sigmoid_scalar(fit.bias_means[k]);
            mean += w * p;
            // dp/dβ = p(1−p); Var(p) ≈ (p(1−p)·std_β)².
            let dp = p * (1.0 - p) * fit.bias_stds[k];
            var += (w * dp) * (w * dp);
        }
        (mean, var)
    };
    let panel_sd = panel_var.sqrt();
    let panel_positional_bias_ci = (
        (panel_positional_bias - z * panel_sd).clamp(0.0, 1.0),
        (panel_positional_bias + z * panel_sd).clamp(0.0, 1.0),
    );

    ScoringResult {
        rankings,
        selection_weights,
        warm_start_state,
        sample_size: 0,
        judge_analytics,
        panel_positional_bias,
        panel_positional_bias_ci,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_judge_info() -> JudgeInfo {
        JudgeInfo {
            judge_ids: vec![42],
            logprobs_mode: true,
        }
    }

    fn dist(p: f64) -> [f64; 4] {
        if p > 0.75 { [1.0, 0.0, 0.0, 0.0] }
        else if p > 0.5 { [0.0, 1.0, 0.0, 0.0] }
        else if p > 0.25 { [0.0, 0.0, 1.0, 0.0] }
        else { [0.0, 0.0, 0.0, 1.0] }
    }

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(id1: i64, id2: i64, prob: f64) -> [ComparisonInput; 2] {
        [
            ComparisonInput { slot1: 0, slot2: 1, item1: id1, item2: id2, category_probs: dist(prob), judge_id: 42 },
            ComparisonInput { slot1: 0, slot2: 1, item1: id2, item2: id1, category_probs: dist(1.0 - prob), judge_id: 42 },
        ]
    }

    fn default_scoring_options() -> ScoringOptions {
        ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            selection_sharpness: None,
            anchor_index: 0.0,
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
    fn test_laplace_and_mcmc_agree_on_ordering() {
        // Both engines fit the same posterior, so on clear data they must
        // produce the same ranking order.
        let item_ids = vec![10, 20, 30];
        let mut comparisons: Vec<ComparisonInput> = Vec::new();
        for _ in 0..10 {
            comparisons.extend(make_pair(10, 20, 0.9));
            comparisons.extend(make_pair(20, 30, 0.9));
            comparisons.extend(make_pair(10, 30, 0.95));
        }
        let ji = single_judge_info();

        let mut mcmc_opts = default_scoring_options();
        mcmc_opts.iterations = 3000;
        mcmc_opts.burn_in = 500;
        mcmc_opts.seed = Some(7);
        let mut laplace_opts = mcmc_opts.clone();
        laplace_opts.inference = InferenceMode::LaplaceLinear;

        let mcmc = run_scoring(&item_ids, &comparisons, &mcmc_opts, &ji);
        let laplace = run_scoring(&item_ids, &comparisons, &laplace_opts, &ji);

        let mcmc_order: Vec<i64> = mcmc.rankings.iter().map(|r| r.item).collect();
        let laplace_order: Vec<i64> = laplace.rankings.iter().map(|r| r.item).collect();
        assert_eq!(mcmc_order, vec![10, 20, 30]);
        assert_eq!(laplace_order, vec![10, 20, 30]);
        // Laplace CIs are finite and bracket the score.
        for r in &laplace.rankings {
            assert!(r.lower_bound <= r.score && r.score <= r.upper_bound);
            assert!(r.lower_bound.is_finite() && r.upper_bound.is_finite());
        }
    }

    #[test]
    fn test_bias_prior_direction() {
        // bias_prior > 0.5 means "judges favor the first-listed item". With NO
        // comparisons the posterior equals the prior, so the reported
        // positional bias must land on the same side of 0.5 as the prior.
        // Regression test for a sign flip where the prior was installed as the
        // cutpoint center without negation, inverting its direction.
        let mut opts = default_scoring_options();
        opts.bias_prior_logit = (0.8_f64 / 0.2).ln(); // bias_prior = 0.8
        opts.iterations = 5000;
        opts.burn_in = 1000;
        // With no likelihood, the center chain samples the prior alone. The
        // production proposal step (0.15) mixes that wide prior too slowly for
        // a stable 5000-sample mean; use a step matched to the prior scale.
        opts.bias_proposal_std = 1.0;

        let ji = single_judge_info();
        let result = run_scoring(&[1, 2], &[], &opts, &ji);
        let bias = result.judge_analytics[0].positional_bias;
        assert!(
            bias > 0.55,
            "bias_prior 0.8 with no data must report positional_bias well above 0.5, got {bias:.4}"
        );
    }

    #[test]
    fn test_cold_start_scoring() {
        let item_ids = vec![100, 200, 300];
        // Clear wins for item 100 (0.95 -> category A) and enough samples that
        // the posterior-mean ordering is stable run to run.
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.95),
            make_pair(100, 300, 0.95),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let mut opts = default_scoring_options();
        opts.iterations = 2000;
        opts.burn_in = 300;

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &opts, &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].item, 100);
        assert!(result.selection_weights.is_none());
        assert_eq!(result.warm_start_state.item_log_strengths.len(), 3);
        assert_eq!(result.sample_size, 2000);
        assert_eq!(result.judge_analytics.len(), 1);
        assert_eq!(result.judge_analytics[0].judge_id, 42);
    }

    #[test]
    fn test_warm_start_scoring() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
            make_pair(20, 30, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result1 = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let mut opts2 = default_scoring_options();
        opts2.warm_start = Some(result1.warm_start_state);
        opts2.burn_in = 0;

        let result2 = run_scoring(&item_ids, &comparisons, &opts2, &ji);

        assert_eq!(result2.rankings.len(), 3);
        assert_eq!(result2.warm_start_state.item_log_strengths.len(), 3);
    }

    #[test]
    fn test_warm_start_state_is_log_strengths_in_both_engines() {
        // Both engines must write the SAME unit into WarmStartState: raw
        // log-strengths (mean-centered θ). Regression test for the Laplace path
        // exp()-ing its means — exp values are all positive, so requiring a
        // negative entry and a ~0 sum catches any strength-space leak.
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = (0..10).flat_map(|_| {
            [make_pair(10, 20, 0.9), make_pair(20, 30, 0.9)]
        }).flatten().collect();
        let ji = single_judge_info();

        for inference in [InferenceMode::Mcmc, InferenceMode::LaplaceLinear] {
            let mut opts = default_scoring_options();
            opts.inference = inference;
            opts.seed = Some(11);
            let state = run_scoring(&item_ids, &comparisons, &opts, &ji).warm_start_state;
            let sum: f64 = state.item_log_strengths.iter().sum();
            assert!(
                sum.abs() < 1e-6,
                "{inference:?}: log-strengths are mean-centered, sum should be ~0, got {sum}"
            );
            assert!(
                state.item_log_strengths.iter().any(|&s| s < 0.0),
                "{inference:?}: the losing item's log-strength must be negative, got {:?}",
                state.item_log_strengths
            );
        }
    }

    #[test]
    #[should_panic(expected = "warm_start item_log_strengths length (2) must match num_items (3)")]
    fn test_warm_start_wrong_length_panics() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let mut opts = default_scoring_options();
        opts.warm_start = Some(WarmStartState {
            item_log_strengths: vec![1.0, 1.0], // Wrong length: 2 instead of 3
            judge_biases: vec![],
        });

        run_scoring(&item_ids, &comparisons, &opts, &ji);
    }

    #[test]
    fn test_scoring_with_selection_weights() {
        let item_ids = vec![1, 2, 3, 4];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(1, 2, 0.9),
            make_pair(1, 3, 0.85),
            make_pair(1, 4, 0.9),
            make_pair(2, 3, 0.7),
            make_pair(2, 4, 0.75),
            make_pair(3, 4, 0.6),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let mut opts = default_scoring_options();
        opts.selection_sharpness = Some(0.5);

        let result = run_scoring(&item_ids, &comparisons, &opts, &ji);

        let weights = result.selection_weights.expect("selection weights requested");
        assert_eq!(weights.len(), 4);
        // The dominant winner (item 1) should carry strictly more selection
        // weight than the clear loser (item 4).
        let idx1 = item_ids.iter().position(|&id| id == 1).unwrap();
        let idx4 = item_ids.iter().position(|&id| id == 4).unwrap();
        assert!(
            weights[idx1] > weights[idx4],
            "leader weight {} should exceed tail weight {}",
            weights[idx1], weights[idx4]
        );
        // Every weight is finite and non-negative.
        assert!(weights.iter().all(|&w| w.is_finite() && w >= 0.0));
    }

    #[test]
    fn test_selection_weights_cutoff_keeps_at_least_two() {
        // A runaway leader with everyone else far behind: the cutoff would zero
        // the whole field, but the top-2 floor must keep two positive weights so
        // a pair can still be drawn.
        let item_ids = vec![1, 2, 3, 4, 5];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(1, 2, 0.99),
            make_pair(1, 3, 0.99),
            make_pair(1, 4, 0.99),
            make_pair(1, 5, 0.99),
            make_pair(2, 3, 0.55),
            make_pair(4, 5, 0.55),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let mut opts = default_scoring_options();
        opts.selection_sharpness = Some(0.5);
        opts.selection_cutoff = 0.5; // aggressive: only the leader clears it on area

        let result = run_scoring(&item_ids, &comparisons, &opts, &ji);
        let weights = result.selection_weights.unwrap();
        let positive = weights.iter().filter(|&&w| w > 0.0).count();
        assert!(positive >= 2, "expected at least two positive weights, got {positive}");
    }

    #[test]
    fn test_scoring_with_arbitrary_ids() {
        let item_ids = vec![999, 42, 7777];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(999, 42, 0.8),
            make_pair(42, 7777, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let ranked_ids: Vec<i64> = result.rankings.iter().map(|r| r.item).collect();
        assert!(ranked_ids.contains(&999));
        assert!(ranked_ids.contains(&42));
        assert!(ranked_ids.contains(&7777));
    }

    #[test]
    #[should_panic(expected = "Unknown item ID")]
    fn test_scoring_unknown_id_panics() {
        let item_ids = vec![1, 2, 3];
        let comparisons = vec![
            ComparisonInput { slot1: 0, slot2: 1, item1: 1, item2: 99, category_probs: dist(0.8), judge_id: 42 },
        ];

        let ji = single_judge_info();
        run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);
    }

    #[test]
    #[should_panic(expected = "at least one judge")]
    fn test_scoring_zero_judges_panics() {
        // An empty judge panel is an illegal state, not a case to handle:
        // the engines would otherwise disagree on the panel bias null value.
        let ji = JudgeInfo { judge_ids: vec![], logprobs_mode: false };
        run_scoring(&[1, 2], &[], &default_scoring_options(), &ji);
    }

    #[test]
    #[should_panic(expected = "Duplicate item ID")]
    fn test_scoring_duplicate_ids_panics() {
        let item_ids = vec![1, 2, 1];

        let ji = single_judge_info();
        run_scoring(&item_ids, &[], &default_scoring_options(), &ji);
    }

    #[test]
    fn test_multi_judge_scoring() {
        let item_ids = vec![100, 200, 300];
        let judge_a = 111;
        let judge_b = 222;

        let comparisons = vec![
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a },
            ComparisonInput { slot1: 0, slot2: 1, item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a },
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b },
            ComparisonInput { slot1: 0, slot2: 1, item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b },
            ComparisonInput { slot1: 0, slot2: 1, item1: 200, item2: 300, category_probs: dist(0.7), judge_id: judge_a },
            ComparisonInput { slot1: 0, slot2: 1, item1: 300, item2: 200, category_probs: dist(0.3), judge_id: judge_b },
        ];

        let ji = JudgeInfo {
            judge_ids: vec![judge_a, judge_b],
            logprobs_mode: true,
        };

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.judge_analytics.len(), 2);
        assert_eq!(result.judge_analytics[0].judge_id, judge_a);
        assert_eq!(result.judge_analytics[1].judge_id, judge_b);
        assert_eq!(result.judge_analytics[0].num_comparisons + result.judge_analytics[1].num_comparisons, 6);
    }

    #[test]
    fn test_single_judge_panel_bias_matches_judge() {
        // With one judge the panel aggregate is that judge's own posterior.
        // The CI must match exactly: quantiles commute with the monotone
        // logit->probability transform, so the same sample ranks are picked.
        let item_ids = vec![100, 200, 300];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let ja = &result.judge_analytics[0];
        assert!((result.panel_positional_bias_ci.0 - ja.positional_bias_ci.0).abs() < 1e-12);
        assert!((result.panel_positional_bias_ci.1 - ja.positional_bias_ci.1).abs() < 1e-12);
        // Point estimates differ only by mean-of-sigmoid vs sigmoid-of-mean.
        assert!((result.panel_positional_bias - ja.positional_bias).abs() < 0.05);
    }

    #[test]
    fn test_multi_judge_panel_bias_within_bounds() {
        let item_ids = vec![100, 200, 300];
        let judge_a = 111;
        let judge_b = 222;

        let comparisons = vec![
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a },
            ComparisonInput { slot1: 0, slot2: 1, item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a },
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b },
            ComparisonInput { slot1: 0, slot2: 1, item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b },
        ];

        let ji = JudgeInfo {
            judge_ids: vec![judge_a, judge_b],
            logprobs_mode: true,
        };

        let mut opts = default_scoring_options();
        opts.iterations = 2000;
        opts.burn_in = 300;
        let result = run_scoring(&item_ids, &comparisons, &opts, &ji);

        let (lo, hi) = result.panel_positional_bias_ci;
        assert!(lo <= result.panel_positional_bias && result.panel_positional_bias <= hi);
        assert!(0.0 < lo && hi < 1.0);
        // The weighted average of independent biases concentrates: the panel CI
        // must not be wider than the widest per-judge CI.
        let max_judge_width = result.judge_analytics.iter()
            .map(|ja| ja.positional_bias_ci.1 - ja.positional_bias_ci.0)
            .fold(0.0_f64, f64::max);
        assert!(hi - lo <= max_judge_width + 1e-12);
    }

    #[test]
    fn test_normal_cdf_known_points() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.959_963_98) - 0.975).abs() < 1e-5); // 97.5% quantile
        assert!((normal_cdf(-1.959_963_98) - 0.025).abs() < 1e-5);
        assert!(normal_cdf(8.0) > 0.999_999);
        assert!(normal_cdf(-8.0) < 1e-6);
        // Symmetry: Φ(x) + Φ(-x) = 1.
        assert!((normal_cdf(1.3) + normal_cdf(-1.3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_selection_weights_leader_and_ordering() {
        // means descending; with anchor_index 0 the leader straddles its own
        // mean (A = 0.5) so its ratio is exactly 1, and weights must decrease
        // with distance below the leader.
        let means = vec![2.0, 1.5, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5, 0.5];
        let games = vec![10, 10, 10, 10];
        // sharpness 1, no cutoff, no coverage pull → pure ratio.
        let w = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        assert!((w[0] - 1.0).abs() < 1e-6, "leader ratio should be 1, got {}", w[0]);
        assert!(w[0] > w[1] && w[1] > w[2] && w[2] > w[3]);
    }

    #[test]
    fn test_target_blend_pulls_toward_prediction_early_then_observed() {
        // 50 items whose observed top (0.5) is modest, but the strength prior
        // (tau2 = 10 → σ₀ ≈ 3.16) predicts a much higher top over 50 draws. When
        // the leader has few games the prediction dominates the target, so even
        // the leader sits confidently below it (ratio ≪ 1). As the leader's game
        // count grows the observed top takes over, the leader straddles the
        // target again, and its ratio returns toward 1.
        let n = 50;
        let mut means = vec![0.0; n];
        means[0] = 0.5; // modest observed top
        let stds = vec![1.0; n];
        let prior_tau2 = 10.0;

        // Few games on the leader → prediction-dominated target → ratio ≪ 1.
        let mut games_few = vec![10usize; n];
        games_few[0] = 2;
        let w_early = compute_selection_weights(&means, &stds, &games_few, 1.0, 0.0, 0.0, 0.0, prior_tau2, 10.0);
        assert!(w_early[0] < 0.45, "early: leader ratio should be pulled well below 1, got {}", w_early[0]);

        // Many games on the leader → observed-dominated target → ratio near 1.
        let mut games_many = vec![10usize; n];
        games_many[0] = 1000;
        let w_late = compute_selection_weights(&means, &stds, &games_many, 1.0, 0.0, 0.0, 0.0, prior_tau2, 10.0);
        assert!(w_late[0] > 0.8, "late: leader ratio should approach 1, got {}", w_late[0]);
        assert!(w_late[0] > w_early[0]);

        // Blend disabled (prior games = 0) → leader exactly at ratio 1 regardless of games.
        let w_off = compute_selection_weights(&means, &stds, &games_few, 1.0, 0.0, 0.0, 0.0, prior_tau2, 0.0);
        assert!((w_off[0] - 1.0).abs() < 1e-8, "blend off: leader ratio should be 1, got {}", w_off[0]);
    }

    #[test]
    fn test_anchor_index_moves_anchor_down_the_ranking() {
        // With anchor_index 1.0 the 2nd-best item becomes the anchor: it sits
        // at ratio 1 (maximum focus), while items equidistant on either side
        // of it get equal, lower weight — the leader is no longer special once
        // it is confidently above the boundary. Blend disabled so the target
        // is the pure observed anchor.
        let means = vec![2.0, 1.5, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5, 0.5];
        let games = vec![10, 10, 10, 10];
        let w = compute_selection_weights(&means, &stds, &games, 1.0, 1.0, 0.0, 0.0, 10.0, 0.0);
        assert!((w[1] - 1.0).abs() < 1e-6, "anchor (rank 1) ratio should be 1, got {}", w[1]);
        assert!(w[1] > w[0], "anchor should outweigh the confident leader");
        assert!(
            (w[0] - w[2]).abs() < 1e-9,
            "items equidistant above/below the anchor should weigh the same: {} vs {}",
            w[0], w[2]
        );
        assert!(w[2] > w[3], "farther below the anchor should weigh less");
    }

    #[test]
    fn test_anchor_index_fractional_interpolates() {
        // anchor_index 0.5 targets the midpoint of the rank-0 and rank-1 means
        // (2.0 and 1.5 → 1.75). The two items sit at ±0.5σ from that target,
        // so both get the identical ratio Φ(−0.5)/Φ(0.5). Also continuous:
        // every weight lies strictly between its anchor-0 and anchor-1 weights.
        let means = vec![2.0, 1.0, 1.5, 0.0];
        let stds = vec![0.5, 0.5, 0.5, 0.5];
        let games = vec![10, 10, 10, 10];
        let w_mid = compute_selection_weights(&means, &stds, &games, 1.0, 0.5, 0.0, 0.0, 10.0, 0.0);
        let expected = normal_cdf(-0.5) / normal_cdf(0.5);
        assert!((w_mid[0] - expected).abs() < 1e-6, "leader ratio should be Φ(−0.5)/Φ(0.5), got {}", w_mid[0]);
        assert!((w_mid[2] - expected).abs() < 1e-6, "rank-1 ratio should be Φ(−0.5)/Φ(0.5), got {}", w_mid[2]);

        let w0 = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        let w1 = compute_selection_weights(&means, &stds, &games, 1.0, 1.0, 0.0, 0.0, 10.0, 0.0);
        for i in 0..means.len() {
            assert!(
                w_mid[i] > w0[i].min(w1[i]) && w_mid[i] < w0[i].max(w1[i]),
                "fractional anchor weight should sit strictly between integer-anchor weights for item {i}"
            );
        }
    }

    #[test]
    fn test_predicted_rank_strength_decreases_with_anchor_index() {
        // The predicted order statistic must decrease monotonically as the
        // anchor moves down the ranking, and anchor 0 must reproduce the
        // classic E[max] plotting position.
        let means = vec![0.0; 20];
        let prior_tau2 = 10.0;
        let p0 = predicted_rank_strength(&means, prior_tau2, 0.0);
        let p_half = predicted_rank_strength(&means, prior_tau2, 0.5);
        let p1 = predicted_rank_strength(&means, prior_tau2, 1.0);
        let p9 = predicted_rank_strength(&means, prior_tau2, 9.0);
        assert!(p0 > p_half && p_half > p1 && p1 > p9);
        let nf = 20.0_f64;
        let expected_max = prior_tau2.sqrt() * inverse_normal_cdf((nf - 0.375) / (nf + 0.25));
        assert!((p0 - expected_max).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "anchor_index")]
    fn test_anchor_index_beyond_last_item_panics() {
        let means = vec![1.0, 0.0];
        let stds = vec![0.5, 0.5];
        let games = vec![10, 10];
        compute_selection_weights(&means, &stds, &games, 1.0, 1.5, 0.0, 0.0, 10.0, 0.0);
    }

    #[test]
    fn test_compute_selection_weights_sharpness_flattens() {
        // Lower sharpness compresses the leader-to-tail ratio.
        let means = vec![2.0, 1.0];
        let stds = vec![1.0, 1.0];
        let games = vec![10, 10];
        let sharp = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        let soft = compute_selection_weights(&means, &stds, &games, 0.5, 0.0, 0.0, 0.0, 10.0, 0.0);
        let ratio_sharp = sharp[0] / sharp[1];
        let ratio_soft = soft[0] / soft[1];
        assert!(ratio_soft < ratio_sharp, "lower sharpness should flatten the ratio");
    }

    #[test]
    fn test_compute_selection_weights_coverage_pull() {
        // Two items with identical posteriors (so identical area), but one has
        // been played far more. With coverage = 0 their weights match; with
        // coverage > 0 the under-played item is boosted above the over-played one.
        let means = vec![1.0, 1.0];
        let stds = vec![1.0, 1.0];
        let games = vec![2, 50]; // item 0 under-served, item 1 over-served

        let no_pull = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        assert!((no_pull[0] - no_pull[1]).abs() < 1e-12, "coverage 0 should ignore games");

        let pull = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 1.0, 10.0, 0.0);
        assert!(
            pull[0] > pull[1],
            "under-served item should outweigh over-served one under coverage pull: {} vs {}",
            pull[0], pull[1]
        );
        // Proportional-fair (coverage 1): weight ratio should be the inverse game
        // ratio, i.e. 50/2 = 25.
        assert!(((pull[0] / pull[1]) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_logprobs_scoring() {
        let item_ids = vec![100, 200, 300];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = JudgeInfo {
            judge_ids: vec![42],
            logprobs_mode: false,
        };

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.judge_analytics.len(), 1);
    }
}
