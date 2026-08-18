/// Laplace Bradley-Terry scoring wrapper.
///
/// One function, one options struct. Pure function — no IO, no state.
/// Items are identified by caller-provided `i64` IDs.
use std::collections::HashMap;

use crate::laplace_bt;
use crate::types::{
    ComparisonInput, IdMap, IndexedComparison, JudgeAnalytics, JudgeInfo,
    RankedItem, ScoringOptions, ScoringResult,
};

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

/// Output of `compute_selection_weights`: the pairing weights plus the raw
/// side-of-anchor statistic that early stopping reads.
struct SelectionWeightsOutput {
    /// Sharpened, cutoff-filtered, coverage-pulled pairing weights.
    weights: Vec<f64>,
    /// ln P(every checked item sits on its posterior-favored side of the
    /// observed anchor): the sum of per-item ln(max(A, 1−A)), treating the
    /// side probabilities as independent (see the comment at the
    /// computation). In (−∞, 0]; the early-stop criterion is
    /// `partition_log_confidence >= ln(c)`. An integer anchor exempts the
    /// anchor item itself (its side probability is ~0.5 by construction); a
    /// fractional anchor exempts nobody, since the boundary sits between two
    /// items that both must resolve. 0.0 only for a single item.
    partition_log_confidence: f64,
}

/// Build top-heavy item-selection weights from each item's posterior summary.
///
/// For each item, estimate which side of the anchor it lands on: with
/// `A = P(strength_i ≥ strength_anchor)` — treating both as independent
/// Gaussian summaries, so `A = Φ((mean_i − target) / sqrt(std_i² + std_anchor²))`
/// with the anchor's own uncertainty widening the split — the base weight is
/// the uncertainty ratio `min(A, 1−A) / max(A, 1−A)`:
/// 1 when the posterior sits astride the anchor (maximally unsure which side
/// the item belongs on), decaying toward 0 as the item resolves confidently
/// above or below it — raised to `selection_sharpness` (lower = flatter =
/// more exploration). The anchor is the item at rank `anchor_index` (0-based,
/// posterior means sorted descending; fractional values interpolate between
/// the two adjacent ranks); the anchor itself sits at ratio 1 once the target
/// blend has converged to the observed anchor (early on, while
/// `target_prior_games` still pulls the target toward the prior prediction,
/// even the anchor can sit below 1). Items whose
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
) -> SelectionWeightsOutput {
    let n = means.len();
    if n == 0 {
        return SelectionWeightsOutput { weights: Vec::new(), partition_log_confidence: 0.0 };
    }
    assert!(
        anchor_index.is_finite() && anchor_index >= 0.0 && anchor_index <= (n - 1) as f64,
        "anchor_index={anchor_index}, must be finite and in [0, num_items - 1 = {}]",
        n - 1
    );

    // The anchor: the item at rank `anchor_index` when posterior means are
    // sorted descending. Fractional indices interpolate linearly between the
    // two adjacent ranks — for the anchor's mean, variance, and game count.
    let mut by_mean: Vec<usize> = (0..n).collect();
    by_mean.sort_by(|&a, &b| means[b].partial_cmp(&means[a]).unwrap_or(std::cmp::Ordering::Equal));
    let lo_rank = anchor_index.floor() as usize;
    let hi_rank = anchor_index.ceil() as usize;
    let frac = anchor_index - lo_rank as f64;
    let (lo_idx, hi_idx) = (by_mean[lo_rank], by_mean[hi_rank]);
    let observed_anchor = means[lo_idx] + frac * (means[hi_idx] - means[lo_idx]);
    let anchor_var = {
        let var_lo = stds[lo_idx] * stds[lo_idx];
        let var_hi = stds[hi_idx] * stds[hi_idx];
        var_lo + frac * (var_hi - var_lo)
    };

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

    // Uncertainty ratio per item: A = P(item above the anchor) under the
    // difference of the two independent Gaussian summaries — the anchor's own
    // variance widens the split, so a still-uncertain anchor keeps nearby items
    // in play; the correction fades as the anchor plays games. The ratio
    // min/max is 1 when the item straddles the anchor and decays toward 0 once
    // it is confidently on either side. max(A, 1−A) >= 0.5, so the division is
    // always safe.
    let ratios: Vec<f64> = (0..n)
        .map(|i| {
            let spread = (stds[i] * stds[i] + anchor_var).sqrt();
            let above = if spread <= 1e-12 {
                // Degenerate point masses: strictly-above the target counts fully,
                // exactly-at counts half, below counts nothing.
                if means[i] > target {
                    1.0
                } else if means[i] >= target {
                    0.5
                } else {
                    0.0
                }
            } else {
                normal_cdf((means[i] - target) / spread)
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

    let mut weights: Vec<f64> = (0..n)
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
        .collect();

    // In an extreme finite-precision state every uncertainty ratio can round
    // to exactly zero, leaving no distribution for top-heavy selection to
    // sample. Keep the run moving by making item selection uniform. This only
    // handles the all-zero case; NaNs and negative weights are not converted by
    // this recovery and remain invalid.
    if weights.iter().all(|&weight| weight == 0.0) {
        weights.fill(1.0);
    }

    // Stopping statistic: ln P(every checked item sits on its
    // posterior-favored side of the anchor), taking the per-item side
    // probabilities max(A, 1−A) as independent and summing their logs (the
    // log domain keeps thousands of near-1 factors from underflowing). Items
    // far from the anchor contribute ~ln(1) = 0, so the sum is dominated by
    // the few straddling the boundary. Independence is optimistic in form but
    // conservative in effect: the side events are positively correlated
    // through the shared anchor, so the product underestimates the joint
    // probability and a stop based on it fires late, not early.
    //
    // Measured against the OBSERVED anchor, not the blended target: the blend
    // deliberately inflates the early target toward the prior-predicted anchor
    // for exploration, and items "confidently below" that inflated boundary
    // are not resolved against the actual anchor — measuring against the blend
    // fires spurious stops in the first rounds.
    //
    // An integer anchor exempts the anchor item itself (some item must hold
    // the boundary; its own side probability is ~0.5 by construction and
    // would floor the product). A fractional anchor exempts NOBODY: the
    // boundary sits between two items, and both neighbours must resolve
    // confidently onto their sides of the virtual target — exempting them
    // would leave the boundary pair unchecked, and with two items would make
    // the stop fire vacuously on the first fit. The only remaining empty sum
    // (a single item) gives ln(1) = 0: vacuously certain.
    let partition_log_confidence = (0..n)
        .filter(|&i| frac > 0.0 || i != lo_idx)
        .map(|i| {
            let spread = (stds[i] * stds[i] + anchor_var).sqrt();
            let above = if spread <= 1e-12 {
                // Degenerate point masses: strictly-above counts fully,
                // exactly-at counts half, below counts nothing.
                if means[i] > observed_anchor {
                    1.0
                } else if means[i] >= observed_anchor {
                    0.5
                } else {
                    0.0
                }
            } else {
                normal_cdf((means[i] - observed_anchor) / spread)
            };
            // Side probability: the favored side's mass, in [0.5, 1] — the
            // log is finite.
            above.max(1.0 - above).ln()
        })
        .sum::<f64>();

    SelectionWeightsOutput { weights, partition_log_confidence }
}

/// Run Laplace Bradley-Terry scoring on pairwise comparison data.
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
/// - `options.selection_sharpness` is set and `options.anchor_index` is not
///   finite or lies outside `[0, item_ids.len() - 1]`
pub fn run_scoring(
    item_ids: &[i64],
    comparisons: &[ComparisonInput],
    options: &ScoringOptions,
    judge_info: &JudgeInfo,
) -> ScoringResult {
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

    build_scoring_result(&id_map, num_items, &indexed, options, judge_info)
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

/// Deterministic MAP fit plus curvature, with confidence intervals computed as
/// Gaussian intervals from the per-parameter standard deviations.
#[allow(clippy::needless_range_loop)]
fn build_scoring_result(
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
    for &(i, j, _, k, _, _, _) in indexed {
        games_per_item[i] += 1;
        games_per_item[j] += 1;
        comparisons_per_judge[k] += 1;
    }

    // Selection weights reuse the shared Gaussian-area helper on (mean, std).
    let selection = options.selection_sharpness.map(|sharpness| {
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
    let (selection_weights, partition_log_confidence) = match selection {
        Some(s) => (Some(s.weights), Some(s.partition_log_confidence)),
        None => (None, None),
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
    let panel_positional_bias_ci = if num_judges == 1 {
        judge_analytics[0].positional_bias_ci
    } else {
        let panel_sd = panel_var.sqrt();
        (
            (panel_positional_bias - z * panel_sd).clamp(0.0, 1.0),
            (panel_positional_bias + z * panel_sd).clamp(0.0, 1.0),
        )
    };

    ScoringResult {
        rankings,
        selection_weights,
        partition_log_confidence,
        item_means: fit.means.clone(),
        item_stds: fit.stds.clone(),
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

    fn dist(p: f64) -> [f64; 2] {
        if p > 0.5 { [1.0, 0.0] } else { [0.0, 1.0] }
    }

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(id1: i64, id2: i64, prob: f64) -> [ComparisonInput; 2] {
        [
            ComparisonInput { slot1: 0, slot2: 1, item1: id1, item2: id2, category_probs: dist(prob), judge_id: 42, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: id2, item2: id1, category_probs: dist(1.0 - prob), judge_id: 42, weight: 1.0 },
        ]
    }

    fn default_scoring_options() -> ScoringOptions {
        ScoringOptions {
            confidence_level: 0.95,
            selection_sharpness: None,
            anchor_index: 0.0,
            selection_cutoff: 0.05,
            selection_coverage: 0.0,
            target_prior_games: 10.0,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            bias_prior_tau2: 2.0,
            bias_prior_logit: 0.0,
        }
    }

    #[test]
    fn test_clear_data_produces_expected_ordering_and_intervals() {
        let item_ids = vec![10, 20, 30];
        let mut comparisons: Vec<ComparisonInput> = Vec::new();
        for _ in 0..10 {
            comparisons.extend(make_pair(10, 20, 0.9));
            comparisons.extend(make_pair(20, 30, 0.9));
            comparisons.extend(make_pair(10, 30, 0.95));
        }
        let ji = single_judge_info();

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);
        let order: Vec<i64> = result.rankings.iter().map(|r| r.item).collect();
        assert_eq!(order, vec![10, 20, 30]);
        for r in &result.rankings {
            assert!(r.lower_bound <= r.score && r.score <= r.upper_bound);
            assert!(r.lower_bound.is_finite() && r.upper_bound.is_finite());
        }
    }

    #[test]
    fn test_item_means_and_stds_are_input_order_posteriors() {
        // item_means/item_stds are the flat per-item posterior summary in
        // item_ids order — rankings is the same data sorted by score. Each
        // ranked score must equal the item_means entry for that item, and every
        // std must be a usable (finite, positive) value.
        let item_ids = vec![10, 20, 30];
        let mut comparisons: Vec<ComparisonInput> = Vec::new();
        for _ in 0..10 {
            comparisons.extend(make_pair(10, 20, 0.9));
            comparisons.extend(make_pair(20, 30, 0.9));
            comparisons.extend(make_pair(10, 30, 0.95));
        }
        let ji = single_judge_info();

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);
        assert_eq!(result.item_means.len(), item_ids.len());
        assert_eq!(result.item_stds.len(), item_ids.len());
        for r in &result.rankings {
            let idx = item_ids.iter().position(|&id| id == r.item).unwrap();
            assert_eq!(r.score, result.item_means[idx]);
        }
        for &s in &result.item_stds {
            assert!(s.is_finite() && s > 0.0, "posterior std must be finite and positive, got {s}");
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

        let ji = single_judge_info();
        let result = run_scoring(&[1, 2], &[], &opts, &ji);
        let bias = result.judge_analytics[0].positional_bias;
        assert!(
            (bias - 0.8).abs() < 1e-10,
            "bias_prior 0.8 with no data must report positional_bias 0.8, got {bias:.4}"
        );
    }

    #[test]
    fn test_scoring() {
        let item_ids = vec![100, 200, 300];
        // Clear wins for item 100 (0.95 -> category A).
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.95),
            make_pair(100, 300, 0.95),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].item, 100);
        assert!(result.selection_weights.is_none());
        assert!(result.partition_log_confidence.is_none());
        assert_eq!(result.judge_analytics.len(), 1);
        assert_eq!(result.judge_analytics[0].judge_id, 42);
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
        // The stopping statistic ships alongside the weights: a finite
        // log-probability in (−∞, 0].
        let log_conf = result.partition_log_confidence.expect("computed with selection weights");
        assert!(log_conf.is_finite() && log_conf <= 0.0, "got {log_conf}");
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
            ComparisonInput { slot1: 0, slot2: 1, item1: 1, item2: 99, category_probs: dist(0.8), judge_id: 42, weight: 1.0 },
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
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 200, item2: 300, category_probs: dist(0.7), judge_id: judge_a, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 300, item2: 200, category_probs: dist(0.3), judge_id: judge_b, weight: 1.0 },
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
        // With one judge the panel aggregate is exactly that judge's estimate
        // and transformed-Gaussian interval.
        let item_ids = vec![100, 200, 300];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let ja = &result.judge_analytics[0];
        assert!((result.panel_positional_bias - ja.positional_bias).abs() < 1e-12);
        assert_eq!(result.panel_positional_bias_ci, ja.positional_bias_ci);
    }

    #[test]
    fn test_multi_judge_panel_bias_within_bounds() {
        let item_ids = vec![100, 200, 300];
        let judge_a = 111;
        let judge_b = 222;

        let comparisons = vec![
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b, weight: 1.0 },
            ComparisonInput { slot1: 0, slot2: 1, item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b, weight: 1.0 },
        ];

        let ji = JudgeInfo {
            judge_ids: vec![judge_a, judge_b],
            logprobs_mode: true,
        };

        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        let (lo, hi) = result.panel_positional_bias_ci;
        assert!(lo <= result.panel_positional_bias && result.panel_positional_bias <= hi);
        assert!(0.0 < lo && hi < 1.0);
        for judge in &result.judge_analytics {
            assert!(
                judge.positional_bias_ci.0 <= judge.positional_bias
                    && judge.positional_bias <= judge.positional_bias_ci.1
            );
        }
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
        let w = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0).weights;
        assert!((w[0] - 1.0).abs() < 1e-6, "leader ratio should be 1, got {}", w[0]);
        assert!(w[0] > w[1] && w[1] > w[2] && w[2] > w[3]);
    }

    #[test]
    fn test_all_zero_selection_weights_become_uniform() {
        // A fractional anchor halfway across a huge gap puts every finite-width
        // posterior so far into a normal tail that the ordinary CDF calculation
        // rounds all uncertainty ratios to zero. Selection must remain defined.
        let means = vec![100.0, 0.0, -100.0];
        let stds = vec![0.01, 0.01, 0.01];
        let games = vec![10, 10, 10];

        let out = compute_selection_weights(
            &means, &stds, &games, 1.0, 0.5, 0.0, 0.0, 10.0, 0.0,
        );

        assert_eq!(out.weights, vec![1.0, 1.0, 1.0]);
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
        let w_early = compute_selection_weights(&means, &stds, &games_few, 1.0, 0.0, 0.0, 0.0, prior_tau2, 10.0).weights;
        assert!(w_early[0] < 0.45, "early: leader ratio should be pulled well below 1, got {}", w_early[0]);

        // Many games on the leader → observed-dominated target → ratio near 1.
        let mut games_many = vec![10usize; n];
        games_many[0] = 1000;
        let w_late = compute_selection_weights(&means, &stds, &games_many, 1.0, 0.0, 0.0, 0.0, prior_tau2, 10.0).weights;
        assert!(w_late[0] > 0.8, "late: leader ratio should approach 1, got {}", w_late[0]);
        assert!(w_late[0] > w_early[0]);

        // Blend disabled (prior games = 0) → leader exactly at ratio 1 regardless of games.
        let w_off = compute_selection_weights(&means, &stds, &games_few, 1.0, 0.0, 0.0, 0.0, prior_tau2, 0.0).weights;
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
        let w = compute_selection_weights(&means, &stds, &games, 1.0, 1.0, 0.0, 0.0, 10.0, 0.0).weights;
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
        // (2.0 and 1.5 → 1.75). The two items sit at ±0.25 from that target,
        // with combined spread √(σ² + σ_anchor²) = √0.5, so both get the
        // identical ratio Φ(−0.25/√0.5)/Φ(0.25/√0.5). Also continuous:
        // every weight lies strictly between its anchor-0 and anchor-1 weights.
        let means = vec![2.0, 1.0, 1.5, 0.0];
        let stds = vec![0.5, 0.5, 0.5, 0.5];
        let games = vec![10, 10, 10, 10];
        let w_mid = compute_selection_weights(&means, &stds, &games, 1.0, 0.5, 0.0, 0.0, 10.0, 0.0).weights;
        let z = 0.25 / 0.5_f64.sqrt();
        let expected = normal_cdf(-z) / normal_cdf(z);
        assert!((w_mid[0] - expected).abs() < 1e-6, "leader ratio should be Φ(−z)/Φ(z), got {}", w_mid[0]);
        assert!((w_mid[2] - expected).abs() < 1e-6, "rank-1 ratio should be Φ(−z)/Φ(z), got {}", w_mid[2]);

        let w0 = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0).weights;
        let w1 = compute_selection_weights(&means, &stds, &games, 1.0, 1.0, 0.0, 0.0, 10.0, 0.0).weights;
        for i in 0..means.len() {
            assert!(
                w_mid[i] > w0[i].min(w1[i]) && w_mid[i] < w0[i].max(w1[i]),
                "fractional anchor weight should sit strictly between integer-anchor weights for item {i}"
            );
        }
    }

    #[test]
    fn test_anchor_std_zero_reduces_to_point_target() {
        // With the anchor's std at 0 the combined spread collapses to the
        // item's own std, so every ratio must equal the plain point-target
        // split Φ((mean − target)/std) — the pre-integration formula.
        let means = vec![2.0, 1.0, 0.0];
        let stds = vec![0.0, 0.8, 0.4]; // anchor (rank 0) is a point mass
        let games = vec![10, 10, 10];
        let w = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0).weights;
        assert!((w[0] - 1.0).abs() < 1e-12, "anchor ratio should be 1, got {}", w[0]);
        for i in 1..3 {
            let a = normal_cdf((means[i] - means[0]) / stds[i]);
            let expected = a.min(1.0 - a) / a.max(1.0 - a);
            assert!(
                (w[i] - expected).abs() < 1e-12,
                "item {i}: expected point-target ratio {expected}, got {}",
                w[i]
            );
        }
    }

    #[test]
    fn test_anchor_uncertainty_keeps_boundary_items_in_play() {
        // Same means and item stds, but an uncertain anchor: an item
        // confidently below a *tight* anchor may still straddle an *uncertain*
        // one, so its ratio must rise with the anchor's std — and anneal back
        // as the anchor tightens. The anchor itself stays at ratio 1.
        let means = vec![2.0, 0.5, 0.0];
        let games = vec![10, 10, 10];
        let tight = compute_selection_weights(
            &means, &[0.1, 0.5, 0.5], &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0,
        ).weights;
        let loose = compute_selection_weights(
            &means, &[1.5, 0.5, 0.5], &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0,
        ).weights;
        assert!((tight[0] - 1.0).abs() < 1e-6, "tight anchor ratio should be 1, got {}", tight[0]);
        assert!((loose[0] - 1.0).abs() < 1e-6, "loose anchor ratio should be 1, got {}", loose[0]);
        assert!(
            loose[1] > tight[1] && loose[2] > tight[2],
            "uncertain anchor should raise below-boundary ratios: {} vs {}, {} vs {}",
            loose[1], tight[1], loose[2], tight[2]
        );
    }

    #[test]
    fn test_partition_log_confidence_sums_non_anchor_side_probs() {
        // Anchor rank 0 (item 0) is exempt — its ~0.5 side probability would
        // floor the product. The statistic is the sum of ln(side prob) over
        // the other items. Blend disabled so the target is the anchor mean.
        let means = vec![2.0, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5];
        let games = vec![10, 10, 10];
        let out = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0);
        let spread = (0.25_f64 + 0.25).sqrt();
        let p1 = normal_cdf((2.0 - 1.0) / spread); // item 1's favored-side mass
        let p2 = normal_cdf((2.0 - 0.0) / spread); // item 2's favored-side mass
        let expected = p1.ln() + p2.ln();
        assert!(
            (out.partition_log_confidence - expected).abs() < 1e-12,
            "expected ln(p1)+ln(p2) = {expected}, got {}",
            out.partition_log_confidence
        );
        assert!(out.partition_log_confidence < 0.0, "less than certain");
        assert!(
            out.partition_log_confidence > 0.5_f64.ln(),
            "anchor's ~0.5 side probability must be excluded from the product"
        );
    }

    #[test]
    fn test_partition_log_confidence_fractional_anchor_checks_neighbours() {
        // A fractional anchor sits BETWEEN two items, so neither neighbour is
        // exempt: the boundary pair itself must resolve against the virtual
        // target. Here the neighbours (2.0 and 1.5) straddle the 1.75 midpoint
        // far more than the distant third item, so they dominate the
        // statistic and keep the stop from firing.
        let means = vec![2.0, 1.5, -3.0];
        let stds = vec![0.5, 0.5, 0.5];
        let games = vec![10, 10, 10];
        let out = compute_selection_weights(&means, &stds, &games, 1.0, 0.5, 0.0, 0.0, 10.0, 0.0);
        let target = 1.75; // midpoint of the two anchor-rank means
        let spread = (0.25_f64 + 0.25).sqrt();
        let p_neighbour = normal_cdf(0.25 / spread); // each neighbour, ±0.25 from the midpoint
        let p_third = normal_cdf((target - -3.0) / spread);
        let expected = 2.0 * p_neighbour.ln() + p_third.ln();
        assert!(
            (out.partition_log_confidence - expected).abs() < 1e-12,
            "expected 2·ln(p_neighbour)+ln(p_third) = {expected}, got {}",
            out.partition_log_confidence
        );
        assert!(
            out.partition_log_confidence < 0.95_f64.ln(),
            "unresolved boundary must block a 95% stop"
        );
    }

    #[test]
    fn test_partition_log_confidence_two_items_fractional_anchor_not_vacuous() {
        // Regression: two items with anchor_index 0.5 used to exempt both,
        // making the statistic an empty sum of 0.0 = ln(1) — an unconditional
        // stop on the first fit. Both items must now resolve against the
        // midpoint boundary.
        let means = vec![1.0, 0.0];
        let stds = vec![0.5, 0.5];
        let games = vec![10, 10];
        let out = compute_selection_weights(&means, &stds, &games, 1.0, 0.5, 0.0, 0.0, 10.0, 0.0);
        let spread = (0.25_f64 + 0.25).sqrt();
        let p = normal_cdf(0.5 / spread); // each item, ±0.5 from the midpoint
        let expected = 2.0 * p.ln();
        assert!(
            (out.partition_log_confidence - expected).abs() < 1e-12,
            "expected 2·ln(p) = {expected}, got {}",
            out.partition_log_confidence
        );
        assert!(
            out.partition_log_confidence < 0.95_f64.ln(),
            "must not read as vacuously resolved"
        );
    }

    #[test]
    fn test_partition_log_confidence_ignores_target_blend() {
        // Regression: early in a run the target blend inflates the selection
        // target toward the prior-predicted anchor, so every item sits
        // "confidently below" a boundary that is not the anchor and the
        // selection ratios collapse. The stopping statistic must measure
        // against the observed anchor instead and stay high — otherwise a
        // confidence-based early stop fires after the first uniform round.
        let n = 50;
        let mut means = vec![0.0; n];
        means[0] = 0.5; // modest observed leader, far below the predicted max
        let stds = vec![1.0; n];
        let mut games = vec![10usize; n];
        games[0] = 2; // few anchor games → blend dominated by the prediction
        let out = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 10.0);
        // Selection sees everyone (leader included) below the blended target.
        assert!(out.weights[0] < 0.45, "sanity: blend should crush the leader's weight, got {}", out.weights[0]);
        // The stopper sees items 1..n straddling the observed anchor (gap 0.5,
        // spread sqrt(2)): 49 items at side probability ~0.64 multiply to a
        // partition confidence near zero — nowhere near any stop threshold.
        assert!(
            out.partition_log_confidence < -1.0,
            "partition confidence must stay far below any stop threshold under a blend-dominated target, got {}",
            out.partition_log_confidence
        );
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
        let sharp = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0).weights;
        let soft = compute_selection_weights(&means, &stds, &games, 0.5, 0.0, 0.0, 0.0, 10.0, 0.0).weights;
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

        let no_pull = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0).weights;
        assert!((no_pull[0] - no_pull[1]).abs() < 1e-12, "coverage 0 should ignore games");

        let pull = compute_selection_weights(&means, &stds, &games, 1.0, 0.0, 0.0, 1.0, 10.0, 0.0).weights;
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
