/// Laplace Bradley-Terry scoring wrapper.
///
/// One function, one options struct. Pure function — no IO, no state.
/// Items are identified by caller-provided `i64` IDs.
use std::collections::HashMap;

use crate::laplace_bt;
use crate::types::{
    Edge, IdMap, IndexedEdge, JudgeAnalytics, JudgeInfo,
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
/// top-k statistic that early stopping reads.
struct SelectionWeightsOutput {
    /// Sharpened, cutoff-filtered, coverage-pulled pairing weights.
    weights: Vec<f64>,
    /// ln P(every item sits on its posterior-favored side of the top-k
    /// boundary): the sum over all items of ln(max(A, 1−A)), where `A` is the
    /// item's chance of being in the top `selection_top_k`, treating the side
    /// probabilities as independent (see the comment at the computation). In
    /// (−∞, 0]; the early-stop criterion is
    /// `partition_log_confidence >= ln(c)`.
    partition_log_confidence: f64,
}

/// Build top-heavy item-selection weights from each item's posterior summary.
///
/// For each item, `A` is its chance of being in the top `selection_top_k`
/// when every item's strength is drawn independently from its posterior
/// `N(mean, std²)` and the whole field is ranked (see
/// `top_k_probabilities`). The base weight is the uncertainty ratio
/// `min(A, 1−A) / max(A, 1−A)`: 1 when the item is as likely in the top k as
/// out of it, decaying toward 0 as it resolves confidently onto either side —
/// raised to `selection_sharpness` (lower = flatter = more exploration). Items
/// whose ratio is below `selection_cutoff` are dropped to 0, except the two
/// highest-ratio items, which are always kept so the pairing layer can always
/// draw two distinct contenders.
///
/// The base weight is then divided by `edge_counts^selection_coverage` — a
/// proportional-fair coverage pull that drives each item's cumulative edge
/// count toward its ratio-implied share (`selection_coverage = 0` disables it, `1`
/// is standard proportional-fair). `edge_counts` uses the current cumulative
/// counts, so a resolved item (ratio → 0) sheds its weight rather than carrying
/// stale "owed" edges.
///
/// Weights are returned unnormalized; the pairing layer normalizes by its
/// running total.
#[allow(clippy::too_many_arguments)]
fn compute_selection_weights(
    means: &[f64],
    stds: &[f64],
    edge_counts: &[usize],
    selection_sharpness: f64,
    selection_top_k: usize,
    selection_cutoff: f64,
    selection_coverage: f64,
    selection_draws: usize,
    selection_seed: u64,
) -> SelectionWeightsOutput {
    let n = means.len();
    if n == 0 {
        return SelectionWeightsOutput { weights: Vec::new(), partition_log_confidence: 0.0 };
    }
    assert!(
        selection_top_k >= 1 && selection_top_k <= n,
        "selection_top_k={selection_top_k}, must be in [1, num_items = {n}]"
    );
    assert!(selection_draws >= 1, "selection_draws={selection_draws}, must be >= 1");

    // A = each item's chance of being in the top k.
    let top_k_chances = top_k_probabilities(means, stds, selection_top_k, selection_draws, selection_seed);

    // Uncertainty ratio per item: min/max of the chance of being in vs out of
    // the top k. 1 when the item is a coin flip for the top k, decaying toward
    // 0 once it is confidently in or out. max(A, 1−A) >= 0.5, so the division
    // is always safe.
    let ratios: Vec<f64> = top_k_chances
        .iter()
        .map(|&above| {
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
            // Proportional-fair coverage: divide by edge_count^selection_coverage. Guard
            // the denominator at 1 so an item with no edges does not blow up
            // (the uniform stage guarantees >= min_uniform_edges before these
            // weights are actually used for pairing).
            let served = (edge_counts[i] as f64).max(1.0);
            base / served.powf(selection_coverage)
        })
        .collect();

    // Every uncertainty ratio can be exactly zero: in an extreme
    // finite-precision state where each chance rounds to 0 or 1, or when
    // `selection_top_k` equals the item count (every item is in the top k with
    // certainty). That leaves no distribution for top-heavy selection to
    // sample. Keep the run moving by making item selection uniform. This only
    // handles the all-zero case; NaNs and negative weights are not converted by
    // this recovery and remain invalid.
    if weights.iter().all(|&weight| weight == 0.0) {
        weights.fill(1.0);
    }

    // Stopping statistic: ln P(every item sits on its posterior-favored side
    // of the top-k boundary), taking the per-item side probabilities
    // max(A, 1−A) as independent and summing their logs (the log domain keeps
    // thousands of near-1 factors from underflowing). Items far from the
    // boundary contribute ~ln(1) = 0, so the sum is dominated by the few
    // contesting it. No item is exempt: the k-th item's chance rises toward 1
    // as it resolves, like any other member of the top k. Order within the top
    // k does not matter, only membership.
    let partition_log_confidence = top_k_chances
        .iter()
        // Side probability: the favored side's mass, in [0.5, 1] — the log is
        // finite.
        .map(|&above| above.max(1.0 - above).ln())
        .sum::<f64>();

    SelectionWeightsOutput { weights, partition_log_confidence }
}

/// One standard normal sample, by the Box-Muller transform (avoids a
/// dependency on `rand_distr`).
fn standard_normal(rng: &mut rand::rngs::StdRng) -> f64 {
    use rand::Rng;
    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Per-item chance of being among the `top_k` strongest when every item's
/// strength is drawn independently from its posterior `N(mean, std²)` and the
/// whole field is ranked. Estimated from `draws` random draws of the field,
/// seeded by `seed`.
///
/// Each draw credits item `i` with its exact probability of beating the bar it
/// has to clear — the `top_k`-th strongest of the OTHER items' drawn values —
/// rather than a 0/1 indicator of whether its own drawn value made the cut. The
/// estimate is therefore smooth: a small chance is never rounded to zero just
/// because the item never made the top k in any draw.
///
/// With `top_k` equal to the item count every item is in the top k with
/// certainty.
fn top_k_probabilities(means: &[f64], stds: &[f64], top_k: usize, draws: usize, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    let n = means.len();
    if top_k >= n {
        return vec![1.0; n];
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut totals = vec![0.0f64; n];
    let mut values = vec![0.0f64; n];
    // The `top_k + 1` largest drawn values, descending. `top_k < n`, so there
    // are always that many.
    let mut strongest: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..draws {
        for i in 0..n {
            values[i] = means[i] + stds[i] * standard_normal(&mut rng);
        }
        strongest.clear();
        strongest.extend_from_slice(&values);
        if top_k + 1 < n {
            strongest.select_nth_unstable_by(top_k, |a, b| b.total_cmp(a));
            strongest.truncate(top_k + 1);
        }
        strongest.sort_by(|a, b| b.total_cmp(a));
        for i in 0..n {
            // The `top_k`-th strongest of the others: if item i's own drawn
            // value is among the top k of the full draw, the others' k-th
            // strongest is one place further down.
            let bar = if values[i] >= strongest[top_k - 1] { strongest[top_k] } else { strongest[top_k - 1] };
            totals[i] += if stds[i] <= 1e-12 {
                // Degenerate point mass: strictly above the bar counts fully,
                // exactly at it counts half, below counts nothing.
                if means[i] > bar {
                    1.0
                } else if means[i] >= bar {
                    0.5
                } else {
                    0.0
                }
            } else {
                normal_cdf((means[i] - bar) / stds[i])
            };
        }
    }
    totals.iter().map(|total| (total / draws as f64).clamp(0.0, 1.0)).collect()
}

/// Run Laplace Bradley-Terry scoring on edge data.
///
/// `item_ids` is the full list of item IDs being ranked. The returned
/// `selection_weights` (when requested) is in the same order as `item_ids`.
///
/// `judge_info.judge_ids` are sorted internally by value so the internal
/// index assignment is deterministic regardless of input order. The returned
/// `judge_analytics` is in ascending `judge_id` order.
///
/// # Panics
///
/// Panics if caller-supplied data violates the input contract:
/// - `judge_info.judge_ids` is empty (a tournament needs at least one judge)
/// - `item_ids` contains a duplicate ID
/// - an edge references an item ID not present in `item_ids`
/// - an edge's `judge_id` is not present in `judge_info.judge_ids`
/// - `options.selection_sharpness` is set and `options.selection_top_k` lies
///   outside `[1, item_ids.len()]`, or `options.selection_draws` is 0
pub fn run_scoring(
    item_ids: &[i64],
    edges: &[Edge],
    options: &ScoringOptions,
    judge_info: &JudgeInfo,
) -> ScoringResult {
    assert!(
        !judge_info.judge_ids.is_empty(),
        "judge_info.judge_ids must contain at least one judge"
    );

    let id_map = IdMap::from_ids(item_ids);
    let num_items = id_map.len();

    // Sort judge ids so internal index assignment is deterministic regardless
    // of the order callers supply them.
    let mut sorted_judge_ids = judge_info.judge_ids.clone();
    sorted_judge_ids.sort();
    let mut judge_id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(sorted_judge_ids.len());
    for (idx, &id) in sorted_judge_ids.iter().enumerate() {
        judge_id_to_idx.insert(id, idx);
    }

    let indexed = id_map.convert_edges(edges, &judge_id_to_idx);

    let sorted_judge_info = JudgeInfo {
        judge_ids: sorted_judge_ids,
        logprobs_mode: judge_info.logprobs_mode,
    };
    build_scoring_result(&id_map, num_items, &indexed, options, &sorted_judge_info)
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
    indexed: &[IndexedEdge],
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

    // Per-item and per-judge edge counts.
    let mut edges_per_item = vec![0usize; num_items];
    let mut edges_per_judge = vec![0usize; num_judges];
    for &(i, j, _, k, _, _, _) in indexed {
        edges_per_item[i] += 1;
        edges_per_item[j] += 1;
        edges_per_judge[k] += 1;
    }

    // Selection weights from each item's posterior summary (mean, std).
    let selection = options.selection_sharpness.map(|sharpness| {
        compute_selection_weights(
            &fit.means,
            &fit.stds,
            &edges_per_item,
            sharpness,
            options.selection_top_k,
            options.selection_cutoff,
            options.selection_coverage,
            options.selection_draws,
            options.selection_seed,
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
                num_edges: edges_per_judge[k],
            }
        })
        .collect();

    // Panel bias: edge-count-weighted average of the judges' bias
    // probabilities (equal weights when there are no edges), with the CI
    // propagated from the per-judge bias variances via the delta method.
    let total_edges: usize = edges_per_judge.iter().sum();
    let (panel_positional_bias, panel_var) = if num_judges == 0 {
        (0.5, 0.0)
    } else {
        let mut mean = 0.0;
        let mut var = 0.0;
        for k in 0..num_judges {
            let w = if total_edges == 0 {
                1.0 / num_judges as f64
            } else {
                edges_per_judge[k] as f64 / total_edges as f64
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
    fn make_pair(id1: i64, id2: i64, prob: f64) -> [Edge; 2] {
        [
            Edge { slot1: 0, slot2: 1, item1: id1, item2: id2, category_probs: dist(prob), judge_id: 42, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: id2, item2: id1, category_probs: dist(1.0 - prob), judge_id: 42, weight: 1.0 },
        ]
    }

    fn default_scoring_options() -> ScoringOptions {
        ScoringOptions {
            confidence_level: 0.95,
            selection_sharpness: None,
            selection_top_k: 1,
            selection_cutoff: 0.05,
            selection_coverage: 0.0,
            selection_draws: 1000,
            selection_seed: 7,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            bias_prior_tau2: 2.0,
            bias_prior_logit: 0.0,
        }
    }

    #[test]
    fn test_clear_data_produces_expected_ordering_and_intervals() {
        let item_ids = vec![10, 20, 30];
        let mut edges: Vec<Edge> = Vec::new();
        for _ in 0..10 {
            edges.extend(make_pair(10, 20, 0.9));
            edges.extend(make_pair(20, 30, 0.9));
            edges.extend(make_pair(10, 30, 0.95));
        }
        let ji = single_judge_info();

        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);
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
        let mut edges: Vec<Edge> = Vec::new();
        for _ in 0..10 {
            edges.extend(make_pair(10, 20, 0.9));
            edges.extend(make_pair(20, 30, 0.9));
            edges.extend(make_pair(10, 30, 0.95));
        }
        let ji = single_judge_info();

        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);
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
        // edges the posterior equals the prior, so the reported
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
        let edges: Vec<Edge> = [
            make_pair(100, 200, 0.95),
            make_pair(100, 300, 0.95),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);

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
        let edges: Vec<Edge> = [
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

        let result = run_scoring(&item_ids, &edges, &opts, &ji);

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
        let edges: Vec<Edge> = [
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

        let result = run_scoring(&item_ids, &edges, &opts, &ji);
        let weights = result.selection_weights.unwrap();
        let positive = weights.iter().filter(|&&w| w > 0.0).count();
        assert!(positive >= 2, "expected at least two positive weights, got {positive}");
    }

    #[test]
    fn test_scoring_with_arbitrary_ids() {
        let item_ids = vec![999, 42, 7777];
        let edges: Vec<Edge> = [
            make_pair(999, 42, 0.8),
            make_pair(42, 7777, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);

        let ranked_ids: Vec<i64> = result.rankings.iter().map(|r| r.item).collect();
        assert!(ranked_ids.contains(&999));
        assert!(ranked_ids.contains(&42));
        assert!(ranked_ids.contains(&7777));
    }

    #[test]
    #[should_panic(expected = "Unknown item ID")]
    fn test_scoring_unknown_id_panics() {
        let item_ids = vec![1, 2, 3];
        let edges = vec![
            Edge { slot1: 0, slot2: 1, item1: 1, item2: 99, category_probs: dist(0.8), judge_id: 42, weight: 1.0 },
        ];

        let ji = single_judge_info();
        run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);
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

        let edges = vec![
            Edge { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 200, item2: 300, category_probs: dist(0.7), judge_id: judge_a, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 300, item2: 200, category_probs: dist(0.3), judge_id: judge_b, weight: 1.0 },
        ];

        let ji = JudgeInfo {
            judge_ids: vec![judge_a, judge_b],
            logprobs_mode: true,
        };

        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.judge_analytics.len(), 2);
        assert_eq!(result.judge_analytics[0].judge_id, judge_a);
        assert_eq!(result.judge_analytics[1].judge_id, judge_b);
        assert_eq!(result.judge_analytics[0].num_edges + result.judge_analytics[1].num_edges, 6);
    }

    #[test]
    fn test_single_judge_panel_bias_matches_judge() {
        // With one judge the panel aggregate is exactly that judge's estimate
        // and transformed-Gaussian interval.
        let item_ids = vec![100, 200, 300];
        let edges: Vec<Edge> = [
            make_pair(100, 200, 0.9),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);

        let ja = &result.judge_analytics[0];
        assert!((result.panel_positional_bias - ja.positional_bias).abs() < 1e-12);
        assert_eq!(result.panel_positional_bias_ci, ja.positional_bias_ci);
    }

    #[test]
    fn test_multi_judge_panel_bias_within_bounds() {
        let item_ids = vec![100, 200, 300];
        let judge_a = 111;
        let judge_b = 222;

        let edges = vec![
            Edge { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b, weight: 1.0 },
            Edge { slot1: 0, slot2: 1, item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b, weight: 1.0 },
        ];

        let ji = JudgeInfo {
            judge_ids: vec![judge_a, judge_b],
            logprobs_mode: true,
        };

        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);

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
    fn test_top_k_probabilities_two_items_match_closed_form() {
        // With two items and k = 1 the chance is P(x0 > x1) =
        // Φ((m0 − m1) / sqrt(s0² + s1²)).
        let means = vec![1.0, 0.0];
        let stds = vec![0.6, 0.8];
        let chances = top_k_probabilities(&means, &stds, 1, 20_000, 11);
        let expected = normal_cdf(1.0 / (0.36f64 + 0.64).sqrt());
        assert!((chances[0] - expected).abs() < 0.01, "got {}, expected {expected}", chances[0]);
        assert!((chances[1] - (1.0 - expected)).abs() < 0.01, "got {}", chances[1]);
    }

    #[test]
    fn test_top_k_probabilities_sum_to_k_and_follow_the_means() {
        let means = vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5];
        let stds = vec![0.7; 6];
        let chances = top_k_probabilities(&means, &stds, 2, 4000, 3);
        assert!(chances.iter().all(|&c| (0.0..=1.0).contains(&c)));
        // Exactly two items are in the top 2 of every draw, so the chances sum
        // to 2 (up to sampling noise).
        let sum: f64 = chances.iter().sum();
        assert!((sum - 2.0).abs() < 0.1, "chances should sum to k = 2, got {sum}");
        for pair in chances.windows(2) {
            assert!(pair[0] > pair[1], "chances should fall with the mean: {chances:?}");
        }
    }

    #[test]
    fn test_top_k_probabilities_small_chance_is_never_zero() {
        // Item 1 is four standard deviations behind: its true chance of being
        // first is about 0.002. Ten draws would almost never see it win, but
        // the smooth estimate still credits it a small positive chance.
        let means = vec![4.0, 0.0];
        let stds = vec![1.0, 1.0];
        let chances = top_k_probabilities(&means, &stds, 1, 10, 5);
        assert!(chances[1] > 0.0 && chances[1] < 0.05, "got {}", chances[1]);
    }

    #[test]
    fn test_top_k_probabilities_seed_reproducible_and_fresh() {
        let means = vec![1.0, 0.5, 0.0, -0.5];
        let stds = vec![0.8; 4];
        let first = top_k_probabilities(&means, &stds, 2, 200, 42);
        let same = top_k_probabilities(&means, &stds, 2, 200, 42);
        let other = top_k_probabilities(&means, &stds, 2, 200, 43);
        assert_eq!(first, same, "same seed must reproduce the same chances");
        assert_ne!(first, other, "a different seed must give different draws");
    }

    #[test]
    fn test_top_k_equal_to_item_count_is_certain_for_everyone() {
        // Every item is in the top n of n: all chances are 1, every ratio is 0,
        // so selection falls back to uniform and the partition is fully
        // resolved.
        let means = vec![1.0, 0.0, -1.0];
        let stds = vec![1.0, 1.0, 1.0];
        let edges = vec![5, 5, 5];
        assert_eq!(top_k_probabilities(&means, &stds, 3, 100, 1), vec![1.0, 1.0, 1.0]);
        let out = compute_selection_weights(&means, &stds, &edges, 1.0, 3, 0.0, 0.0, 100, 1);
        assert_eq!(out.weights, vec![1.0, 1.0, 1.0]);
        assert_eq!(out.partition_log_confidence, 0.0);
    }

    #[test]
    fn test_all_zero_selection_weights_become_uniform() {
        // Point-mass posteriors in a strict order: each item is in the top 1
        // with chance exactly 1 or 0, so every uncertainty ratio is zero.
        // Selection must remain defined.
        let means = vec![3.0, 2.0, 1.0];
        let stds = vec![0.0, 0.0, 0.0];
        let edges = vec![10, 10, 10];

        let out = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 0.0, 50, 9);

        assert_eq!(out.weights, vec![1.0, 1.0, 1.0]);
        assert_eq!(out.partition_log_confidence, 0.0);
    }

    #[test]
    fn test_selection_top_k_moves_the_focus_down_the_ranking() {
        // Ten evenly spaced items. Aiming at first place, the weight sits on
        // the two items contesting it; aiming at the top 5, it sits on the 5th
        // and 6th items, which contest that boundary.
        let means: Vec<f64> = (0..10).map(|i| (9 - i) as f64).collect();
        let stds = vec![0.5; 10];
        let edges = vec![10; 10];
        let heaviest_two = |top_k: usize| {
            let weights = compute_selection_weights(&means, &stds, &edges, 1.0, top_k, 0.0, 0.0, 2000, 17).weights;
            let mut order: Vec<usize> = (0..10).collect();
            order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
            let mut top = vec![order[0], order[1]];
            top.sort();
            top
        };
        assert_eq!(heaviest_two(1), vec![0, 1]);
        assert_eq!(heaviest_two(5), vec![4, 5]);
    }

    #[test]
    fn test_selection_weights_fall_away_from_the_boundary() {
        // Aiming at first place: the two front-runners are close to a coin
        // flip for it, and weight falls with distance behind them.
        let means = vec![2.0, 1.9, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5, 0.5];
        let edges = vec![10, 10, 10, 10];
        let w = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 0.0, 4000, 23).weights;
        assert!(w[0] > w[2] && w[1] > w[2] && w[2] > w[3], "got {w:?}");
        assert!(w[3] > 0.0, "a distant item keeps a small positive weight");
    }

    #[test]
    fn test_selection_cutoff_drops_all_but_the_two_highest_ratios() {
        let means = vec![2.0, 1.9, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5, 0.5];
        let edges = vec![10, 10, 10, 10];
        let w = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.99, 0.0, 4000, 23).weights;
        assert!(w[0] > 0.0 && w[1] > 0.0, "the two highest ratios are always kept: {w:?}");
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);
    }

    #[test]
    fn test_partition_log_confidence_sums_every_items_side_probability() {
        let means = vec![2.0, 1.2, 1.0, 0.0, -1.0];
        let stds = vec![0.4, 0.5, 0.6, 0.5, 0.4];
        let edges = vec![10; 5];
        let (top_k, draws, seed) = (2, 500, 31);
        let out = compute_selection_weights(&means, &stds, &edges, 1.0, top_k, 0.0, 0.0, draws, seed);
        let expected: f64 = top_k_probabilities(&means, &stds, top_k, draws, seed)
            .iter()
            .map(|&above| above.max(1.0 - above).ln())
            .sum();
        assert!((out.partition_log_confidence - expected).abs() < 1e-12);
        assert!(out.partition_log_confidence < 0.0);
    }

    #[test]
    fn test_partition_log_confidence_tied_pair_is_two_coin_flips() {
        // Two identical point masses contesting first place: each is in the
        // top 1 with chance exactly 0.5, and both count — nobody is exempt.
        let means = vec![1.0, 1.0];
        let stds = vec![0.0, 0.0];
        let edges = vec![10, 10];
        let out = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 0.0, 20, 1);
        assert!((out.partition_log_confidence - 2.0 * 0.5f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_partition_log_confidence_rises_as_the_boundary_resolves() {
        // Same means; tighter posteriors leave less doubt about who is in the
        // top 2.
        let means = vec![3.0, 2.0, 1.0, 0.0];
        let edges = vec![10; 4];
        let loose = compute_selection_weights(&means, &[1.0; 4], &edges, 1.0, 2, 0.0, 0.0, 2000, 13);
        let tight = compute_selection_weights(&means, &[0.1; 4], &edges, 1.0, 2, 0.0, 0.0, 2000, 13);
        assert!(tight.partition_log_confidence > loose.partition_log_confidence);
        assert!(tight.partition_log_confidence > 0.999f64.ln());
    }

    #[test]
    #[should_panic(expected = "selection_top_k")]
    fn test_selection_top_k_beyond_item_count_panics() {
        let means = vec![2.0, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5];
        let edges = vec![10, 10, 10];
        compute_selection_weights(&means, &stds, &edges, 1.0, 4, 0.0, 0.0, 100, 1);
    }

    #[test]
    #[should_panic(expected = "selection_top_k")]
    fn test_selection_top_k_zero_panics() {
        let means = vec![2.0, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5];
        let edges = vec![10, 10, 10];
        compute_selection_weights(&means, &stds, &edges, 1.0, 0, 0.0, 0.0, 100, 1);
    }

    #[test]
    #[should_panic(expected = "selection_draws")]
    fn test_selection_draws_zero_panics() {
        let means = vec![2.0, 1.0, 0.0];
        let stds = vec![0.5, 0.5, 0.5];
        let edges = vec![10, 10, 10];
        compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 0.0, 0, 1);
    }

    #[test]
    fn test_compute_selection_weights_sharpness_flattens() {
        // Lower sharpness compresses the contender-to-tail ratio.
        let means = vec![1.0, 0.8, -1.0];
        let stds = vec![1.0, 1.0, 1.0];
        let edges = vec![10, 10, 10];
        let sharp = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 0.0, 2000, 3).weights;
        let soft = compute_selection_weights(&means, &stds, &edges, 0.5, 1, 0.0, 0.0, 2000, 3).weights;
        let ratio_sharp = sharp[0] / sharp[2];
        let ratio_soft = soft[0] / soft[2];
        assert!(ratio_sharp > 1.0);
        assert!(ratio_soft < ratio_sharp, "lower sharpness should flatten the ratio");
    }

    #[test]
    fn test_compute_selection_weights_coverage_pull() {
        // Two identical point masses (each exactly a coin flip for first
        // place, so identical ratios), but one has been accumulated far more.
        // With coverage = 0 their weights match; with coverage > 0 the
        // under-accumulated item is boosted above the over-accumulated one.
        let means = vec![1.0, 1.0];
        let stds = vec![0.0, 0.0];
        let edges = vec![2, 50]; // item 0 under-served, item 1 over-served

        let no_pull = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 0.0, 20, 1).weights;
        assert!((no_pull[0] - no_pull[1]).abs() < 1e-12, "coverage 0 should ignore edges");

        let pull = compute_selection_weights(&means, &stds, &edges, 1.0, 1, 0.0, 1.0, 20, 1).weights;
        assert!(
            pull[0] > pull[1],
            "under-served item should outweigh over-served one under coverage pull: {} vs {}",
            pull[0], pull[1]
        );
        // Proportional-fair (coverage 1): weight ratio should be the inverse edge
        // ratio, i.e. 50/2 = 25.
        assert!(((pull[0] / pull[1]) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_logprobs_scoring() {
        let item_ids = vec![100, 200, 300];
        let edges: Vec<Edge> = [
            make_pair(100, 200, 0.9),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = JudgeInfo {
            judge_ids: vec![42],
            logprobs_mode: false,
        };

        let result = run_scoring(&item_ids, &edges, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.judge_analytics.len(), 1);
    }
}
