/// Unified MCMC scoring wrapper.
///
/// One function, one options struct. Pure function — no IO, no state.
/// Items are identified by caller-provided `i64` IDs.
use crate::gaussian_bt::GaussianBT;
use crate::types::{ComparisonInput, IdMap, ScoringOptions, ScoringResult};

/// Run MCMC scoring on pairwise comparison data.
///
/// `item_ids` is the full list of item IDs being ranked. The returned `state`,
/// `top_k_probs`, and `sample_means` are in the same order as `item_ids`.
pub fn run_scoring(
    item_ids: &[i64],
    comparisons: &[ComparisonInput],
    options: &ScoringOptions,
) -> ScoringResult {
    let id_map = IdMap::from_ids(item_ids);
    let num_items = id_map.len();
    let indexed = id_map.convert_comparisons(comparisons);

    let mut mcmc = GaussianBT::new(
        num_items,
        &indexed,
        options.regularization_strength,
    );

    let samples_result = if let Some(ref warm_start) = options.warm_start {
        assert_eq!(
            warm_start.len(), num_items,
            "warm_start length ({}) must match num_items ({})",
            warm_start.len(), num_items
        );
        mcmc.calculate_incremental_with_samples(warm_start, options.iterations, options.burn_in, options.top_k)
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

    // Convert bias samples from logit space to probability space for confidence interval
    let bias_samples = &samples_result.bias_logit_samples;
    let bias_prob = 1.0 / (1.0 + (-samples_result.bias_logit_mean).exp());
    let bias_confidence_interval = if bias_samples.is_empty() {
        (0.5, 0.5)
    } else {
        let alpha = 1.0 - options.confidence_level;
        let n = bias_samples.len();
        let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n as f64).floor() as usize;
        let upper_idx = upper_idx.saturating_sub(1).max(lower_idx);
        let lower = 1.0 / (1.0 + (-bias_samples[lower_idx]).exp());
        let upper = 1.0 / (1.0 + (-bias_samples[upper_idx]).exp());
        (lower, upper)
    };

    ScoringResult {
        rankings,
        top_k_probs: if options.top_k > 0 { samples_result.top_k_probs } else { None },
        sample_means: if options.top_k > 0 { Some(samples_result.means) } else { None },
        state: mcmc.get_current_state(),
        sample_size: options.iterations,
        positional_bias: bias_prob,
        positional_bias_confidence_interval: bias_confidence_interval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns both position orders for a matchup. In production, the pairing
    /// code's 50/50 coin flip achieves this naturally.
    fn make_pair(id1: i64, id2: i64, prob: f64) -> [ComparisonInput; 2] {
        [
            ComparisonInput { item1: id1, item2: id2, item1_win_probability: prob },
            ComparisonInput { item1: id2, item2: id1, item1_win_probability: 1.0 - prob },
        ]
    }

    #[test]
    fn test_cold_start_scoring() {
        let item_ids = vec![100, 200, 300];
        // Both orders per matchup so bias estimation doesn't confound with item strength.
        // In production, the pairing code's 50/50 coin flip handles this naturally.
        let comparisons: Vec<ComparisonInput> = [
            make_pair(100, 200, 0.9),
            make_pair(100, 300, 0.8),
            make_pair(200, 300, 0.7),
        ].into_iter().flatten().collect();

        let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
        });

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].item, 100); // Strongest item first
        assert!(result.top_k_probs.is_none());
        assert_eq!(result.state.len(), 3);
        assert_eq!(result.sample_size, 200);
    }

    #[test]
    fn test_warm_start_scoring() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
            make_pair(20, 30, 0.7),
        ].into_iter().flatten().collect();

        let result1 = run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 100,
            burn_in: 50,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
        });

        let result2 = run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 100,
            burn_in: 0,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: Some(result1.state),
            regularization_strength: 0.01,
        });

        assert_eq!(result2.rankings.len(), 3);
        assert_eq!(result2.state.len(), 3);
    }

    #[test]
    #[should_panic(expected = "warm_start length (2) must match num_items (3)")]
    fn test_warm_start_wrong_length_panics() {
        let item_ids = vec![10, 20, 30];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(10, 20, 0.9),
        ].into_iter().flatten().collect();

        run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 100,
            burn_in: 50,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: Some(vec![1.0, 1.0]), // Wrong length: 2 instead of 3
            regularization_strength: 0.01,
        });
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

        let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            top_k: 2,
            warm_start: None,
            regularization_strength: 0.01,
        });

        assert!(result.top_k_probs.is_some());
        assert_eq!(result.top_k_probs.as_ref().unwrap().len(), 4);
        assert!(result.sample_means.is_some());
    }

    #[test]
    fn test_scoring_with_arbitrary_ids() {
        // Non-contiguous, non-sequential IDs
        let item_ids = vec![999, 42, 7777];
        let comparisons: Vec<ComparisonInput> = [
            make_pair(999, 42, 0.8),
            make_pair(42, 7777, 0.7),
        ].into_iter().flatten().collect();

        let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 100,
            burn_in: 50,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
        });

        // All ranking items should use our IDs
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
            ComparisonInput { item1: 1, item2: 99, item1_win_probability: 0.8 }, // 99 is not in item_ids
        ];

        run_scoring(&item_ids, &comparisons, &ScoringOptions {
            iterations: 100,
            burn_in: 50,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
        });
    }

    #[test]
    #[should_panic(expected = "Duplicate item ID")]
    fn test_scoring_duplicate_ids_panics() {
        let item_ids = vec![1, 2, 1]; // duplicate

        run_scoring(&item_ids, &[], &ScoringOptions {
            iterations: 100,
            burn_in: 50,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
        });
    }

    /// Generate ground-truth BT strengths and comparisons.
    ///
    /// Strengths are 2^z where z ~ N(0,1), normalized to geometric mean = 1.
    /// Win rates follow the BT formula: P(i beats j) = s_i / (s_i + s_j).
    fn generate_ground_truth(n: usize, seed: u64) -> (Vec<i64>, Vec<f64>, Vec<ComparisonInput>) {
        use rand::{SeedableRng, Rng, rngs::SmallRng};

        let item_ids: Vec<i64> = (1..=n as i64).collect();
        let mut rng = SmallRng::seed_from_u64(seed);

        // Box-Muller for N(0,1) samples, then 2^z for BT strengths.
        let mut true_strengths = Vec::with_capacity(n);
        for _ in 0..n {
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            true_strengths.push(2.0_f64.powf(z));
        }

        // Normalize so geometric mean = 1 (same convention as BT scoring).
        let log_mean = true_strengths.iter().map(|s| s.ln()).sum::<f64>() / n as f64;
        for s in &mut true_strengths {
            *s /= log_mean.exp();
        }

        let mut comparisons = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let prob = true_strengths[i] / (true_strengths[i] + true_strengths[j]);
                comparisons.extend(make_pair(item_ids[i], item_ids[j], prob));
            }
        }

        (item_ids, true_strengths, comparisons)
    }

    /// RMSE between estimated and true strengths in log space.
    fn log_rmse(rankings: &[crate::types::RankedItem], true_by_id: &std::collections::HashMap<i64, f64>) -> f64 {
        let n = rankings.len();
        let sum_sq: f64 = rankings.iter().map(|r| {
            let diff = true_by_id[&r.item].ln() - r.score.ln();
            diff * diff
        }).sum();
        (sum_sq / n as f64).sqrt()
    }

    /// Benchmark: compare cold-start vs BT MLE warm-start MCMC against known ground truth.
    ///
    /// Runs 50 trials with random BT strengths, measures log-space RMSE for three
    /// approaches: cold start (500 burn-in), warm start (50 burn-in), warm start
    /// (0 burn-in). At 5000 trials, warm-start with 0 burn-in is best (0.0901)
    /// because BT MLE already starts at the mode, so burn-in just wastes samples.
    #[test]
    fn test_bt_warm_start_vs_cold_start_accuracy() {
        let num_trials = 50;
        let n = 10;
        let iterations = 2000;

        let mut cold_rmses = Vec::with_capacity(num_trials);
        let mut warm50_rmses = Vec::with_capacity(num_trials);
        let mut warm0_rmses = Vec::with_capacity(num_trials);

        for trial in 0..num_trials {
            let seed = 1000 + trial as u64 * 7;
            let (item_ids, true_strengths, comparisons) = generate_ground_truth(n, seed);

            let true_by_id: std::collections::HashMap<i64, f64> = item_ids.iter()
                .zip(true_strengths.iter())
                .map(|(&id, &s)| (id, s))
                .collect();

            let cold = run_scoring(&item_ids, &comparisons, &ScoringOptions {
                iterations,
                burn_in: 500,
                confidence_level: 0.95,
                top_k: 0,
                warm_start: None,
                regularization_strength: 0.01,
            });

            let id_map = IdMap::from_ids(&item_ids);
            let indexed = id_map.convert_comparisons(&comparisons);
            let mut bt = crate::bradley_terry::BradleyTerry::new(n, &indexed, 0.01);
            bt.calculate_scores(30);
            let bt_scores: Vec<f64> = (0..n).map(|i| bt.get_score(i)).collect();

            let warm50 = run_scoring(&item_ids, &comparisons, &ScoringOptions {
                iterations,
                burn_in: 50,
                confidence_level: 0.95,
                top_k: 0,
                warm_start: Some(bt_scores.clone()),
                regularization_strength: 0.01,
            });

            let warm0 = run_scoring(&item_ids, &comparisons, &ScoringOptions {
                iterations,
                burn_in: 0,
                confidence_level: 0.95,
                top_k: 0,
                warm_start: Some(bt_scores),
                regularization_strength: 0.01,
            });

            cold_rmses.push(log_rmse(&cold.rankings, &true_by_id));
            warm50_rmses.push(log_rmse(&warm50.rankings, &true_by_id));
            warm0_rmses.push(log_rmse(&warm0.rankings, &true_by_id));
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let cold_mean = mean(&cold_rmses);
        let warm50_mean = mean(&warm50_rmses);
        let warm0_mean = mean(&warm0_rmses);

        eprintln!("\n=== BT warm-start accuracy benchmark ({num_trials} trials, {n} items, {iterations} MCMC iter) ===");
        eprintln!("Cold   (500 burn-in): mean RMSE = {cold_mean:.4}");
        eprintln!("Warm50  (50 burn-in): mean RMSE = {warm50_mean:.4}");
        eprintln!("Warm0    (0 burn-in): mean RMSE = {warm0_mean:.4}");

        // All three should recover true strengths well (typical RMSE ~0.08-0.11).
        assert!(cold_mean < 0.2, "Cold mean RMSE {cold_mean:.4} too high");
        assert!(warm50_mean < 0.2, "Warm50 mean RMSE {warm50_mean:.4} too high");
        assert!(warm0_mean < 0.2, "Warm0 mean RMSE {warm0_mean:.4} too high");
    }
}
