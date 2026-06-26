/// Ranking engine orchestrator.
///
/// Adapted for a pure computation crate — no async, no HTTP, no IO.
/// The caller performs comparisons externally, then feeds results back.
///
/// Items are identified by caller-provided `i64` IDs.
use crate::bradley_terry::BradleyTerry;
use crate::constants::INITIAL_BRADLEY_TERRY_RATING;
use crate::pairing::{
    generate_uniform_pairings_indexed, generate_top_heavy_pairings_indexed,
    get_effective_comparison_distribution, ComparisonDistribution,
};
use crate::seed::make_rng;
use crate::types::{ComparisonInput, IdMap, Pair};
use rand::rngs::StdRng;

/// Collapse a categorical verdict distribution into a scalar P(item1 wins) for the
/// Bradley-Terry matchmaking fit. A and B count as a win, C and D as a loss.
fn matchmaking_win_prob(probs: &[f64; 4]) -> f64 {
    probs[0] + probs[1]
}

/// Configuration for the ranking engine.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineConfig {
    pub comparison_distribution: ComparisonDistribution,
    pub matchmaking_sharpness: f64,
    pub min_uniform_games: usize,
    pub seed: Option<u64>,
}

pub struct RankingEngine {
    /// Maps between caller i64 IDs and internal 0..N indices.
    id_map: IdMap,

    /// All successful comparisons (stored with caller IDs).
    pub completed_comparisons: Vec<ComparisonInput>,
    /// Games played per item (indexed internally 0..num_items).
    pub games_played: Vec<usize>,
    /// Number of times each item was placed in position 1 of a comparison
    /// prompt (indexed internally 0..num_items). Used by the pairing layer
    /// to balance position assignments across rounds.
    pub first_position_count: Vec<usize>,

    /// Current BT ratings (indexed internally 0..num_items).
    current_ratings: Vec<f64>,

    /// Per-item selection weights for top-heavy pairing (indexed 0..num_items,
    /// same order as item_ids). Caller MUST set this before calling
    /// `generate_pairs_for_round()` when using the TopHeavy distribution past
    /// warm-up. The engine will panic if missing.
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
            completed_comparisons: Vec::new(),
            games_played: vec![0; num_items],
            first_position_count: vec![0; num_items],
            current_ratings: vec![INITIAL_BRADLEY_TERRY_RATING; num_items],
            selection_weights: None,
            config,
            rng,
        }
    }

    /// Number of items being ranked.
    pub fn num_items(&self) -> usize {
        self.id_map.len()
    }

    /// Generate pairs for a round. Returns pairs of item IDs.
    ///
    /// # Panics
    ///
    /// Panics if the round's effective distribution is top-heavy and the MCMC
    /// data is missing or malformed. The effective distribution is top-heavy
    /// when the engine is configured with `ComparisonDistribution::TopHeavy`
    /// and every item has reached `min_uniform_games` (warm-up rounds use
    /// uniform pairing and need no MCMC data). In a top-heavy round
    /// `selection_weights` must be set, with one entry per item.
    pub fn generate_pairs_for_round(&mut self, _round_index: usize) -> Vec<Pair> {
        let num_items = self.id_map.len();
        let pairs_count = calculate_pairs_for_round(num_items);

        let effective_comparison_distribution = get_effective_comparison_distribution(
            self.config.comparison_distribution,
            num_items,
            &self.games_played,
            self.config.min_uniform_games,
        );

        let index_pairs = match effective_comparison_distribution {
            ComparisonDistribution::Uniform => generate_uniform_pairings_indexed(
                num_items,
                pairs_count,
                &self.current_ratings,
                self.config.matchmaking_sharpness,
                &self.first_position_count,
                &self.games_played,
                &mut self.rng,
            ),
            ComparisonDistribution::TopHeavy => {
                let selection_weights = self.selection_weights.as_ref()
                    .expect("TopHeavy distribution requires selection_weights to be set before generating pairs");

                generate_top_heavy_pairings_indexed(
                    num_items,
                    pairs_count,
                    selection_weights,
                    &self.first_position_count,
                    &self.games_played,
                    &mut self.rng,
                )
            }
        };

        // Convert index pairs to ID pairs
        index_pairs.into_iter().map(|(a, b)| {
            (self.id_map.to_id(a), self.id_map.to_id(b))
        }).collect()
    }

    /// Record comparison results from a round.
    ///
    /// # Panics
    ///
    /// Panics if a result references an item ID that was not in the
    /// `item_ids` the engine was created with.
    pub fn record_results(&mut self, results: &[ComparisonInput]) {
        for result in results {
            self.completed_comparisons.push(*result);
            let idx1 = self.id_map.to_idx(result.item1);
            let idx2 = self.id_map.to_idx(result.item2);
            self.games_played[idx1] += 1;
            self.games_played[idx2] += 1;
            self.first_position_count[idx1] += 1;
        }
    }

    /// Update current rating estimates using Bradley-Terry MLE.
    /// BT MLE is judge-agnostic — it just wants (item1, item2, probability) triples.
    pub fn update_current_ratings(&mut self) {
        if self.completed_comparisons.is_empty() {
            return;
        }

        let num_items = self.id_map.len();
        let indexed: Vec<(usize, usize, f64)> = self.completed_comparisons.iter().map(|c| {
            (self.id_map.to_idx(c.item1), self.id_map.to_idx(c.item2), matchmaking_win_prob(&c.category_probs))
        }).collect();
        let mut bt = BradleyTerry::new(num_items, &indexed, 0.01);
        bt.calculate_scores(30);

        for i in 0..num_items {
            self.current_ratings[i] = bt.get_score(i).ln();
        }
    }

    pub fn current_ratings(&self) -> &[f64] {
        &self.current_ratings
    }

    pub fn completed_comparison_count(&self) -> usize {
        self.completed_comparisons.len()
    }
}

/// Calculate pairs for a single round: every item gets compared once.
pub fn calculate_pairs_for_round(num_items: usize) -> usize {
    num_items / 2
}

/// Calculate total expected comparisons across all rounds.
pub fn calculate_total_expected_comparisons(num_items: usize, number_of_rounds: usize) -> usize {
    calculate_pairs_for_round(num_items) * number_of_rounds
}

/// Calculate rounds needed to reach target comparisons.
pub fn calculate_rounds_for_target_comparisons(num_items: usize, target_comparisons: usize) -> usize {
    let pairs_per_round = calculate_pairs_for_round(num_items);
    if pairs_per_round == 0 || target_comparisons == 0 {
        return 0;
    }
    target_comparisons.div_ceil(pairs_per_round)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_pairs_for_round() {
        assert_eq!(calculate_pairs_for_round(1000), 500);
        assert_eq!(calculate_pairs_for_round(10), 5);
        assert_eq!(calculate_pairs_for_round(79), 39);
        assert_eq!(calculate_pairs_for_round(1), 0);
        assert_eq!(calculate_pairs_for_round(0), 0);
    }

    #[test]
    fn test_calculate_total_expected_comparisons() {
        assert_eq!(calculate_total_expected_comparisons(10, 5), 25);
        assert_eq!(calculate_total_expected_comparisons(78, 10), 390);
        assert_eq!(calculate_total_expected_comparisons(1000, 3), 1500);
    }

    #[test]
    fn test_calculate_rounds_for_target() {
        assert_eq!(calculate_rounds_for_target_comparisons(100, 500), 10);
        assert_eq!(calculate_rounds_for_target_comparisons(100, 501), 11);
        assert_eq!(calculate_rounds_for_target_comparisons(100, 0), 0);
        assert_eq!(calculate_rounds_for_target_comparisons(1, 100), 0);
    }

    fn make_input(id1: i64, id2: i64, prob: f64) -> ComparisonInput {
        let category_probs = if prob > 0.75 { [1.0, 0.0, 0.0, 0.0] }
            else if prob > 0.5 { [0.0, 1.0, 0.0, 0.0] }
            else if prob > 0.25 { [0.0, 0.0, 1.0, 0.0] }
            else { [0.0, 0.0, 0.0, 1.0] };
        ComparisonInput { item1: id1, item2: id2, category_probs, judge_id: 0 }
    }

    #[test]
    fn test_engine_basic_workflow() {
        let item_ids = vec![10, 20, 30, 40];
        let config = EngineConfig {
            comparison_distribution: ComparisonDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_games: 3,
            seed: None,
        };

        let mut engine = RankingEngine::new(&item_ids, config);

        let pairs = engine.generate_pairs_for_round(0);
        assert!(!pairs.is_empty());

        // Pairs should contain our IDs, not indices
        for (a, b) in &pairs {
            assert!(item_ids.contains(a), "ID {} not in item_ids", a);
            assert!(item_ids.contains(b), "ID {} not in item_ids", b);
        }

        let results: Vec<ComparisonInput> = pairs.iter()
            .map(|(a, b)| make_input(*a, *b, 0.7))
            .collect();

        engine.record_results(&results);
        engine.update_current_ratings();

        assert_eq!(engine.completed_comparison_count(), pairs.len());
    }

    #[test]
    #[should_panic(expected = "at least two items")]
    fn test_engine_requires_two_items() {
        let config = EngineConfig {
            comparison_distribution: ComparisonDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_games: 3,
            seed: None,
        };
        let _ = RankingEngine::new(&[1], config);
    }

    #[test]
    fn test_first_position_balancing_converges() {
        // Drive the engine through many rounds and verify each item's
        // first-position ratio stays close to 0.5. A pure coin flip would also
        // pass with these sample sizes; this test guards against regressions
        // that would degrade balancing (e.g. accidentally stop tracking).
        let item_ids: Vec<i64> = (1..=20).collect();
        let config = EngineConfig {
            comparison_distribution: ComparisonDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_games: 3,
            seed: None,
        };

        let mut engine = RankingEngine::new(&item_ids, config);

        for round in 0..342 {
            let pairs = engine.generate_pairs_for_round(round);
            let results: Vec<ComparisonInput> = pairs.iter()
                .map(|(a, b)| make_input(*a, *b, 0.5))
                .collect();
            engine.record_results(&results);
        }

        let total_comparisons: usize = engine.games_played.iter().sum::<usize>() / 2;
        assert_eq!(total_comparisons, 3420);

        for i in 0..engine.num_items() {
            let games = engine.games_played[i];
            assert!(games > 0, "item {} played zero games", i);
            let ratio = engine.first_position_count[i] as f64 / games as f64;
            assert!((ratio - 0.5).abs() < 0.10,
                "item {} drifted: first {} / {} = {:.3}",
                i, engine.first_position_count[i], games, ratio);
        }
    }

    #[test]
    #[should_panic(expected = "Duplicate item ID")]
    fn test_engine_rejects_duplicate_ids() {
        let config = EngineConfig {
            comparison_distribution: ComparisonDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_games: 3,
            seed: None,
        };
        let _ = RankingEngine::new(&[1, 2, 1], config);
    }
}
