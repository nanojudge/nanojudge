/// Unified MCMC scoring wrapper.
///
/// One function, one options struct. Pure function — no IO, no state.
/// Items are identified by caller-provided `i64` IDs.
use std::collections::HashMap;

use crate::gaussian_bt::GaussianBT;
use crate::types::{
    ComparisonInput, IdMap, JudgeAnalytics, JudgeInfo, ScoringOptions, ScoringResult,
    WarmStartState,
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

/// Run MCMC scoring on pairwise comparison data.
///
/// `item_ids` is the full list of item IDs being ranked. The returned state,
/// `top_k_probs`, and `sample_means` are in the same order as `item_ids`.
pub fn run_scoring(
    item_ids: &[i64],
    comparisons: &[ComparisonInput],
    options: &ScoringOptions,
    judge_info: &JudgeInfo,
) -> ScoringResult {
    let id_map = IdMap::from_ids(item_ids);
    let num_items = id_map.len();

    // Build judge_id -> internal index mapping
    let mut judge_id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(judge_info.judge_ids.len());
    for (idx, &id) in judge_info.judge_ids.iter().enumerate() {
        judge_id_to_idx.insert(id, idx);
    }

    let indexed = id_map.convert_comparisons(comparisons, &judge_id_to_idx);

    let mut mcmc = GaussianBT::new(
        num_items,
        &indexed,
        options,
        judge_info,
    );

    let samples_result = if let Some(ref warm_start) = options.warm_start {
        assert_eq!(
            warm_start.item_strengths.len(), num_items,
            "warm_start item_strengths length ({}) must match num_items ({})",
            warm_start.item_strengths.len(), num_items
        );
        mcmc.calculate_incremental_with_samples(
            &warm_start.item_strengths,
            &warm_start.judge_biases,
            &warm_start.judge_log_decisiveness,
            &judge_id_to_idx,
            options.iterations,
            options.burn_in,
            options.top_k,
        )
    } else {
        mcmc.calculate_with_samples(options.iterations, options.burn_in, options.top_k)
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

        // Decisiveness is no longer a separate parameter in the ordinal model
        // (subsumed into the per-judge cutpoint spacing).
        let (decisiveness, decisiveness_ci) = (None, None);

        judge_analytics.push(JudgeAnalytics {
            judge_id,
            positional_bias: bias_prob,
            positional_bias_ci: bias_ci,
            decisiveness,
            decisiveness_ci,
            num_comparisons: samples_result.comparisons_per_judge[j],
        });
    }

    // Build warm start state
    let warm_start_state = WarmStartState {
        item_strengths: mcmc.get_current_state(),
        judge_biases: mcmc.get_current_biases(judge_info),
        judge_log_decisiveness: mcmc.get_current_log_decisiveness(judge_info),
    };

    ScoringResult {
        rankings,
        top_k_probs: if options.top_k > 0 { samples_result.top_k_probs } else { None },
        sample_means: if options.top_k > 0 { Some(samples_result.means) } else { None },
        warm_start_state,
        sample_size: options.iterations,
        judge_analytics,
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
    fn make_pair(id1: i64, id2: i64, prob: f64) -> [ComparisonInput; 2] {
        [
            ComparisonInput { item1: id1, item2: id2, category_probs: dist(prob), judge_id: 42 },
            ComparisonInput { item1: id2, item2: id1, category_probs: dist(1.0 - prob), judge_id: 42 },
        ]
    }

    fn default_scoring_options() -> ScoringOptions {
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
    fn test_cold_start_scoring() {
        let item_ids = vec![100, 200, 300];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(100, 300, 0.8),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let result = run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].item, 100);
        assert!(result.top_k_probs.is_none());
        assert_eq!(result.warm_start_state.item_strengths.len(), 3);
        assert_eq!(result.sample_size, 200);
        assert_eq!(result.judge_analytics.len(), 1);
        assert_eq!(result.judge_analytics[0].judge_id, 42);
        assert!(result.judge_analytics[0].decisiveness.is_none());
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
        assert_eq!(result2.warm_start_state.item_strengths.len(), 3);
    }

    #[test]
    #[should_panic(expected = "warm_start item_strengths length (2) must match num_items (3)")]
    fn test_warm_start_wrong_length_panics() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
        ].into_iter().flatten().collect();

        let ji = single_judge_info();
        let mut opts = default_scoring_options();
        opts.warm_start = Some(WarmStartState {
            item_strengths: vec![1.0, 1.0], // Wrong length: 2 instead of 3
            judge_biases: vec![],
            judge_log_decisiveness: vec![],
        });

        run_scoring(&item_ids, &comparisons, &opts, &ji);
    }

    #[test]
    fn test_scoring_with_top_k() {
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
        opts.top_k = 2;

        let result = run_scoring(&item_ids, &comparisons, &opts, &ji);

        assert!(result.top_k_probs.is_some());
        assert_eq!(result.top_k_probs.as_ref().unwrap().len(), 4);
        assert!(result.sample_means.is_some());
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
            ComparisonInput { item1: 1, item2: 99, category_probs: dist(0.8), judge_id: 42 },
        ];

        let ji = single_judge_info();
        run_scoring(&item_ids, &comparisons, &default_scoring_options(), &ji);
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
            ComparisonInput { item1: 100, item2: 200, category_probs: dist(0.9), judge_id: judge_a },
            ComparisonInput { item1: 200, item2: 100, category_probs: dist(0.1), judge_id: judge_a },
            ComparisonInput { item1: 100, item2: 300, category_probs: dist(0.8), judge_id: judge_b },
            ComparisonInput { item1: 300, item2: 100, category_probs: dist(0.2), judge_id: judge_b },
            ComparisonInput { item1: 200, item2: 300, category_probs: dist(0.7), judge_id: judge_a },
            ComparisonInput { item1: 300, item2: 200, category_probs: dist(0.3), judge_id: judge_b },
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
        assert!(result.judge_analytics[0].decisiveness.is_none());
        assert!(result.judge_analytics[1].decisiveness.is_none());
        assert_eq!(result.judge_analytics[0].num_comparisons + result.judge_analytics[1].num_comparisons, 6);
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
        assert!(result.judge_analytics[0].decisiveness.is_none());
        assert!(result.judge_analytics[0].decisiveness_ci.is_none());
    }
}
