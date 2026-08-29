/// Ranking engine orchestrator.
///
/// Adapted for a pure computation crate — no async, no HTTP, no IO.
/// The caller obtains judgements externally, then feeds their edges back.
///
/// Items are identified by caller-provided `i64` IDs.
use crate::bradley_terry::BradleyTerry;
use crate::constants::INITIAL_BRADLEY_TERRY_RATING;
use crate::pairing::{
    assert_lineup_size, generate_uniform_pairings_indexed,
    generate_top_heavy_pairings_indexed, generate_uniform_lineups_indexed,
    generate_top_heavy_lineups_indexed, get_effective_judgement_distribution,
    JudgementDistribution,
};
use crate::seed::make_rng;
use crate::types::{Edge, IdMap, Pair, Lineup};
use rand::rngs::StdRng;

/// P(item1 wins) from a verdict distribution, for the Bradley-Terry
/// matchmaking fit.
fn matchmaking_win_prob(probs: &[f64; 2]) -> f64 {
    probs[0]
}

/// Configuration for the ranking engine.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineConfig {
    pub judgement_distribution: JudgementDistribution,
    pub matchmaking_sharpness: f64,
    pub min_uniform_edges: usize,
    /// Ghost-player regularization for the interim BT MLE refits that drive
    /// uniform-mode matchmaking. Set it to the same value passed to
    /// `run_scoring` so pairing and scoring regularize alike.
    pub regularization_strength: f64,
    pub seed: Option<u64>,
}

pub struct RankingEngine {
    /// Maps between caller i64 IDs and internal 0..N indices.
    id_map: IdMap,

    /// All successful edges (stored with caller IDs).
    pub completed_edges: Vec<Edge>,
    /// Incident edge count per item (indexed internally 0..num_items).
    pub edge_counts: Vec<usize>,
    /// Number of edges on which each item was `item1` (indexed internally
    /// 0..num_items). Used by the pairing layer to balance position assignments
    /// across refits.
    pub item1_edge_counts: Vec<usize>,

    /// Current BT ratings (indexed internally 0..num_items).
    current_ratings: Vec<f64>,

    /// Posterior stds of `current_ratings`, present only when the ratings came
    /// from an interim posterior fit via `set_current_posterior()`. When set,
    /// opponent matchmaking integrates the win probability over this
    /// uncertainty (`calculate_integrated_info_gain`); when `None` (MLE
    /// ratings) the plug-in info gain is used.
    current_stds: Option<Vec<f64>>,

    /// Per-item selection weights for top-heavy pairing (indexed 0..num_items,
    /// same order as item_ids). Caller MUST set this before calling
    /// `generate_pairs()` when using the TopHeavy distribution past
    /// the uniform stage. The engine will panic if missing.
    pub selection_weights: Option<Vec<f64>>,

    config: EngineConfig,
    rng: StdRng,
}

impl RankingEngine {
    /// Create an engine for ranking the given items.
    ///
    /// # Panics
    ///
    /// - if `item_ids` contains a duplicate ID
    /// - if there are fewer than two items
    pub fn new(item_ids: &[i64], config: EngineConfig) -> Self {
        let id_map = IdMap::from_ids(item_ids);
        let num_items = id_map.len();
        assert!(num_items >= 2, "RankingEngine requires at least two items to compare.");

        let rng = make_rng(config.seed, crate::seed::SUBSYSTEM_PAIRING);

        RankingEngine {
            id_map,
            completed_edges: Vec::new(),
            edge_counts: vec![0; num_items],
            item1_edge_counts: vec![0; num_items],
            current_ratings: vec![INITIAL_BRADLEY_TERRY_RATING; num_items],
            current_stds: None,
            selection_weights: None,
            config,
            rng,
        }
    }

    /// Number of items being ranked.
    pub fn num_items(&self) -> usize {
        self.id_map.len()
    }

    /// The judgement distribution the next generated set will use, given the
    /// edge counts so far. Top-heavy only once every item has reached
    /// `min_uniform_edges` (otherwise uniform pairing).
    pub fn effective_distribution(&self) -> JudgementDistribution {
        get_effective_judgement_distribution(
            self.config.judgement_distribution,
            self.id_map.len(),
            &self.edge_counts,
            self.config.min_uniform_edges,
        )
    }

    /// Generate exactly `pairs_count` pairs using the current effective
    /// distribution. Uniform generation balances cumulative incident edge
    /// counts even when the request is too small to include every item or large
    /// enough to include every item several times.
    pub fn generate_pairs(&mut self, pairs_count: usize) -> Vec<Pair> {
        let num_items = self.id_map.len();

        let effective_judgement_distribution = self.effective_distribution();

        let index_pairs = match effective_judgement_distribution {
            JudgementDistribution::Uniform => generate_uniform_pairings_indexed(
                num_items,
                pairs_count,
                &self.current_ratings,
                self.current_stds.as_deref(),
                self.config.matchmaking_sharpness,
                &self.item1_edge_counts,
                &self.edge_counts,
                &mut self.rng,
            ),
            JudgementDistribution::TopHeavy => {
                let selection_weights = self.selection_weights.as_ref()
                    .expect("TopHeavy distribution requires selection_weights to be set before generating pairs");

                generate_top_heavy_pairings_indexed(
                    num_items,
                    pairs_count,
                    selection_weights,
                    &self.current_ratings,
                    self.current_stds.as_deref(),
                    self.config.matchmaking_sharpness,
                    &self.item1_edge_counts,
                    &self.edge_counts,
                    &mut self.rng,
                )
            }
        };

        // Convert index pairs to ID pairs
        index_pairs.into_iter().map(|(a, b)| {
            (self.id_map.to_id(a), self.id_map.to_id(b))
        }).collect()
    }

    /// Generate `lineups_count` lineups of `lineup_size` items each,
    /// using the current effective distribution. The lineup analogue of
    /// `generate_pairs`. Each lineup receives one judgement, then is folded into
    /// edges by the caller via `lineup::winner_dist_to_edges` before being fed
    /// back through `record_edges`.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as `generate_pairs`: top-heavy generation
    /// with `selection_weights` unset or malformed.
    pub fn generate_lineups(&mut self, lineups_count: usize, lineup_size: usize) -> Vec<Lineup> {
        assert_lineup_size(lineup_size);
        let num_items = self.id_map.len();

        let effective_judgement_distribution = self.effective_distribution();

        let index_lineups = match effective_judgement_distribution {
            JudgementDistribution::Uniform => generate_uniform_lineups_indexed(
                num_items,
                lineups_count,
                lineup_size,
                &self.current_ratings,
                self.current_stds.as_deref(),
                self.config.matchmaking_sharpness,
                &self.edge_counts,
                &mut self.rng,
            ),
            JudgementDistribution::TopHeavy => {
                let selection_weights = self.selection_weights.as_ref()
                    .expect("TopHeavy distribution requires selection_weights to be set before generating lineups");

                generate_top_heavy_lineups_indexed(
                    num_items,
                    lineups_count,
                    lineup_size,
                    selection_weights,
                    &self.current_ratings,
                    self.current_stds.as_deref(),
                    self.config.matchmaking_sharpness,
                    &mut self.rng,
                )
            }
        };

        // Convert index lineups to ID lineups.
        index_lineups
            .into_iter()
            .map(|lineup| lineup.into_iter().map(|idx| self.id_map.to_id(idx)).collect())
            .collect()
    }

    /// Record edges collected before a refit.
    ///
    /// # Panics
    ///
    /// Panics if a result references an item ID that was not in the
    /// `item_ids` the engine was created with.
    pub fn record_edges(&mut self, results: &[Edge]) {
        for result in results {
            self.completed_edges.push(*result);
            let idx1 = self.id_map.to_idx(result.item1);
            let idx2 = self.id_map.to_idx(result.item2);
            self.edge_counts[idx1] += 1;
            self.edge_counts[idx2] += 1;
            self.item1_edge_counts[idx1] += 1;
        }
    }

    /// Update current rating estimates using Bradley-Terry MLE.
    /// BT MLE is judge-agnostic — it consumes edges as
    /// `(item1, item2, probability)` tuples.
    pub fn update_current_ratings(&mut self) {
        if self.completed_edges.is_empty() {
            return;
        }

        let num_items = self.id_map.len();
        let indexed: Vec<(usize, usize, f64)> = self.completed_edges.iter().map(|c| {
            (self.id_map.to_idx(c.item1), self.id_map.to_idx(c.item2), matchmaking_win_prob(&c.category_probs))
        }).collect();
        let mut bt = BradleyTerry::new(num_items, &indexed, self.config.regularization_strength);
        bt.calculate_scores(30);

        for i in 0..num_items {
            self.current_ratings[i] = bt.get_score(i).ln();
        }
        // MLE point estimates carry no posterior uncertainty; any stds from an
        // earlier posterior fit would be stale against these fresh ratings.
        self.current_stds = None;
    }

    /// Replace the rating state with an interim posterior summary: per-item
    /// posterior mean log-strengths and stds, both indexed like the `item_ids`
    /// the engine was built with (`ScoringResult::item_means` /
    /// `ScoringResult::item_stds` are already in that order). While set, the
    /// stds make opponent matchmaking integrate win probabilities over the
    /// rating uncertainty instead of using point estimates.
    ///
    /// # Panics
    ///
    /// Panics if `means` or `stds` length does not equal the item count.
    pub fn set_current_posterior(&mut self, means: &[f64], stds: &[f64]) {
        let num_items = self.id_map.len();
        assert_eq!(means.len(), num_items, "means length mismatch");
        assert_eq!(stds.len(), num_items, "stds length mismatch");
        self.current_ratings.copy_from_slice(means);
        self.current_stds = Some(stds.to_vec());
    }

    pub fn current_ratings(&self) -> &[f64] {
        &self.current_ratings
    }

    pub fn current_stds(&self) -> Option<&[f64]> {
        self.current_stds.as_deref()
    }

    pub fn completed_edge_count(&self) -> usize {
        self.completed_edges.len()
    }
}

/// Maximum judgement-attempt budget for a run:
/// `ceil(judgements_per_item * num_items / lineup_size)`.
/// Rounding up ensures the budget contains at least the requested number of
/// item appearances. Each attempt judges one lineup of `lineup_size` items.
///
/// # Panics
///
/// Panics if `lineup_size` is outside the supported range of 2 to 9, if
/// `num_items < lineup_size`, or if `judgements_per_item * num_items`
/// overflows `usize`.
pub fn calculate_budget(num_items: usize, judgements_per_item: usize, lineup_size: usize) -> usize {
    assert_lineup_size(lineup_size);
    assert!(
        num_items >= lineup_size,
        "need at least {lineup_size} items to fill a lineup of that size, got {num_items}"
    );
    judgements_per_item
        .checked_mul(num_items)
        .expect("judgement budget calculation overflow")
        .div_ceil(lineup_size)
}

/// The number of judgements needed for every item to appear at least once.
/// When the item count is not divisible by `lineup_size`, the final judgement
/// contains some items that already appeared. Returns zero when there are too
/// few items to form even one lineup.
///
/// # Panics
///
/// Panics if `lineup_size` is outside the supported range of 2 to 9.
pub fn judgements_needed_for_every_item_to_appear_once(
    num_items: usize,
    lineup_size: usize,
) -> usize {
    assert_lineup_size(lineup_size);
    if num_items < lineup_size {
        0
    } else {
        num_items.div_ceil(lineup_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_budget() {
        // 10 items, 4 judgements per item, pairwise → 4*10/2 = 20
        assert_eq!(calculate_budget(10, 4, 2), 20);
        // 9 items, 6 judgements per item, lineup of 3 → 6*9/3 = 18
        assert_eq!(calculate_budget(9, 6, 3), 18);
        // Exact division: 11 items, 2 jpi, pairwise → 2*11/2 = 11
        assert_eq!(calculate_budget(11, 2, 2), 11);
        // Non-divisible item appearances round up so no requested slot is lost.
        assert_eq!(calculate_budget(10, 1, 3), 4);
        assert_eq!(calculate_budget(11, 1, 2), 6);
    }

    #[test]
    fn test_judgements_needed_for_every_item_to_appear_once() {
        assert_eq!(judgements_needed_for_every_item_to_appear_once(10, 2), 5);
        assert_eq!(judgements_needed_for_every_item_to_appear_once(9, 3), 3);
        assert_eq!(judgements_needed_for_every_item_to_appear_once(11, 2), 6);
        assert_eq!(judgements_needed_for_every_item_to_appear_once(100, 9), 12);
        assert_eq!(judgements_needed_for_every_item_to_appear_once(2, 3), 0);
    }

    #[test]
    #[should_panic(expected = "lineup_size must be between 2 and 9, got 0")]
    fn test_calculate_budget_rejects_zero_lineup_size() {
        let _ = calculate_budget(10, 4, 0);
    }

    #[test]
    #[should_panic(expected = "need at least 9 items to fill a lineup of that size, got 5")]
    fn test_calculate_budget_rejects_too_few_items() {
        let _ = calculate_budget(5, 10, 9);
    }

    #[test]
    #[should_panic(expected = "judgement budget calculation overflow")]
    fn test_calculate_budget_reports_overflow() {
        let _ = calculate_budget(usize::MAX, 2, 2);
    }

    #[test]
    #[should_panic(expected = "lineup_size must be between 2 and 9, got 10")]
    fn test_judgements_needed_for_every_item_to_appear_once_rejects_unsupported_size() {
        let _ = judgements_needed_for_every_item_to_appear_once(10, 10);
    }

    fn make_input(id1: i64, id2: i64, prob: f64) -> Edge {
        let category_probs = if prob > 0.5 { [1.0, 0.0] } else { [0.0, 1.0] };
        Edge { slot1: 0, slot2: 1, item1: id1, item2: id2, category_probs, judge_id: 0, weight: 1.0 }
    }

    #[test]
    fn test_engine_basic_workflow() {
        let item_ids = vec![10, 20, 30, 40];
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };

        let mut engine = RankingEngine::new(&item_ids, config);

        let pairs = engine.generate_pairs(judgements_needed_for_every_item_to_appear_once(
            item_ids.len(),
            2,
        ));
        assert!(!pairs.is_empty());

        // Pairs should contain our IDs, not indices
        for (a, b) in &pairs {
            assert!(item_ids.contains(a), "ID {} not in item_ids", a);
            assert!(item_ids.contains(b), "ID {} not in item_ids", b);
        }

        let results: Vec<Edge> = pairs.iter()
            .map(|(a, b)| make_input(*a, *b, 0.7))
            .collect();

        engine.record_edges(&results);
        engine.update_current_ratings();

        assert_eq!(engine.completed_edge_count(), pairs.len());
    }

    #[test]
    fn test_uniform_pairs_honor_arbitrary_requested_count() {
        let item_ids: Vec<i64> = (0..6).collect();
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 2,
            regularization_strength: 0.01,
            seed: Some(7),
        };
        let mut engine = RankingEngine::new(&item_ids, config);

        let pairs = engine.generate_pairs(11);

        assert_eq!(pairs.len(), 11);
        assert!(pairs.iter().all(|(a, b)| a != b));
    }

    #[test]
    fn test_subdivided_uniform_pairs_reach_edge_threshold_before_advancing() {
        let item_ids: Vec<i64> = (0..5).collect();
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::TopHeavy,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 1,
            regularization_strength: 0.01,
            seed: Some(11),
        };
        let mut engine = RankingEngine::new(&item_ids, config);

        for _ in 0..judgements_needed_for_every_item_to_appear_once(item_ids.len(), 2) {
            let pairs = engine.generate_pairs(1);
            assert_eq!(pairs.len(), 1);
            engine.record_edges(&[make_input(pairs[0].0, pairs[0].1, 0.5)]);
        }

        assert!(engine.edge_counts.iter().all(|&count| count >= 1));
        assert_eq!(engine.effective_distribution(), JudgementDistribution::TopHeavy);
    }

    #[test]
    fn test_uniform_lineups_honor_arbitrary_size_and_cover_remainder() {
        let item_ids: Vec<i64> = (0..10).collect();
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 2,
            regularization_strength: 0.01,
            seed: Some(13),
        };
        let mut engine = RankingEngine::new(&item_ids, config);

        let lineups = engine.generate_lineups(9, 3);

        assert_eq!(lineups.len(), 9);
        assert!(lineups.iter().all(|lineup| {
            lineup.len() == 3
                && lineup[0] != lineup[1]
                && lineup[0] != lineup[2]
                && lineup[1] != lineup[2]
        }));
        let initially_scheduled_items: std::collections::HashSet<i64> = lineups
            [..judgements_needed_for_every_item_to_appear_once(item_ids.len(), 3)]
            .iter()
            .flatten()
            .copied()
            .collect();
        assert_eq!(initially_scheduled_items.len(), item_ids.len());
    }

    #[test]
    fn test_set_current_posterior_and_mle_refit_clears_stds() {
        let item_ids: Vec<i64> = vec![10, 20, 30];
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };
        let mut engine = RankingEngine::new(&item_ids, config);

        let means = vec![0.4, -0.1, 0.9];
        let stds = vec![0.5, 1.2, 0.3];
        engine.set_current_posterior(&means, &stds);
        assert_eq!(engine.current_ratings(), means.as_slice());
        assert_eq!(engine.current_stds(), Some(stds.as_slice()));

        // An MLE refit replaces the ratings, so the posterior stds must not
        // survive it — stale stds against fresh point estimates would be wrong.
        engine.record_edges(&[make_input(10, 20, 0.7)]);
        engine.update_current_ratings();
        assert!(engine.current_stds().is_none());
    }

    #[test]
    #[should_panic(expected = "stds length mismatch")]
    fn test_set_current_posterior_length_mismatch_panics() {
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };
        let mut engine = RankingEngine::new(&[1, 2, 3], config);
        engine.set_current_posterior(&[0.0, 0.0, 0.0], &[1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "at least two items")]
    fn test_engine_requires_two_items() {
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };
        let _ = RankingEngine::new(&[1], config);
    }

    #[test]
    #[should_panic(expected = "lineup_size must be between 2 and 9, got 10")]
    fn test_generate_lineups_rejects_oversized_lineup() {
        // The size is rejected here, before any judge is called, rather than
        // later when the returned lineup fails to fold into edges.
        let item_ids: Vec<i64> = (1..=20).collect();
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };
        let mut engine = RankingEngine::new(&item_ids, config);
        let _ = engine.generate_lineups(2, 10);
    }

    #[test]
    #[should_panic(expected = "lineup_size must be between 2 and 9, got 1")]
    fn test_generate_lineups_rejects_undersized_lineup() {
        let item_ids: Vec<i64> = (1..=20).collect();
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };
        let mut engine = RankingEngine::new(&item_ids, config);
        let _ = engine.generate_lineups(2, 1);
    }

    #[test]
    fn test_first_position_balancing_converges() {
        // Drive the engine through many refits and verify each item's
        // first-position ratio stays close to 0.5. A pure coin flip would also
        // pass with these sample sizes; this test guards against regressions
        // that would degrade balancing (e.g. accidentally stop tracking).
        let item_ids: Vec<i64> = (1..=20).collect();
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };

        let mut engine = RankingEngine::new(&item_ids, config);

        let judgements_to_include_every_item_once =
            judgements_needed_for_every_item_to_appear_once(item_ids.len(), 2);
        for _ in 0..342 {
            let pairs = engine.generate_pairs(judgements_to_include_every_item_once);
            let results: Vec<Edge> = pairs.iter()
                .map(|(a, b)| make_input(*a, *b, 0.5))
                .collect();
            engine.record_edges(&results);
        }

        let total_edges: usize = engine.edge_counts.iter().sum::<usize>() / 2;
        assert_eq!(total_edges, 3420);

        for i in 0..engine.num_items() {
            let edges = engine.edge_counts[i];
            assert!(edges > 0, "item {} accumulated zero edges", i);
            let ratio = engine.item1_edge_counts[i] as f64 / edges as f64;
            assert!((ratio - 0.5).abs() < 0.10,
                "item {} drifted: first {} / {} = {:.3}",
                i, engine.item1_edge_counts[i], edges, ratio);
        }
    }

    #[test]
    #[should_panic(expected = "Duplicate item ID")]
    fn test_engine_rejects_duplicate_ids() {
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            regularization_strength: 0.01,
            seed: None,
        };
        let _ = RankingEngine::new(&[1, 2, 1], config);
    }
}
