/// Ranking engine orchestrator.
///
/// Adapted for a pure computation crate — no async, no HTTP, no IO.
/// The caller performs comparisons externally, then feeds results back.
///
/// Items are identified by caller-provided `i64` IDs.
use crate::bradley_terry::BradleyTerry;
use crate::constants::{CONCURRENCY_LIMIT, INITIAL_BRADLEY_TERRY_RATING, MAX_ROUND_MULTIPLIER};
use crate::pairing::{
    generate_balanced_pairings_indexed, generate_top_heavy_pairings_indexed,
    get_effective_strategy, Strategy,
};
use crate::types::{ComparisonInput, IdMap, Pair};

/// Configuration for the ranking engine.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineConfig {
    pub strategy: Strategy,
    pub matchmaking_sharpness: f64,
    pub min_games_before_strategy: usize,
    pub number_of_rounds: Option<usize>,
}

pub struct RankingEngine {
    /// Maps between caller i64 IDs and internal 0..N indices.
    id_map: IdMap,

    /// All successful comparisons (stored with caller IDs).
    pub completed_comparisons: Vec<ComparisonInput>,
    /// Games played per item (indexed internally 0..num_items).
    pub games_played: Vec<usize>,

    /// Current BT ratings (indexed internally 0..num_items).
    current_ratings: Vec<f64>,

    current_round_number: usize,

    /// MCMC data for top-heavy pairing (indexed 0..num_items, same order as item_ids).
    /// Caller MUST set these before calling `generate_pairs_for_round()` when using
    /// TopHeavy strategy. The engine will panic if missing.
    pub mcmc_sample_means: Option<Vec<f64>>,
    pub mcmc_top_k_probs: Option<Vec<f64>>,

    config: EngineConfig,
}

impl RankingEngine {
    pub fn new(item_ids: &[i64], config: EngineConfig) -> Self {
        let id_map = IdMap::from_ids(item_ids);
        let num_items = id_map.len();
        assert!(num_items >= 2, "RankingEngine requires at least two items to compare.");

        RankingEngine {
            id_map,
            completed_comparisons: Vec::new(),
            games_played: vec![0; num_items],
            current_ratings: vec![INITIAL_BRADLEY_TERRY_RATING; num_items],
            current_round_number: 0,
            mcmc_sample_means: None,
            mcmc_top_k_probs: None,
            config,
        }
    }

    /// Number of items being ranked.
    pub fn num_items(&self) -> usize {
        self.id_map.len()
    }

    /// Generate pairs for a round. Returns pairs of item IDs.
    pub fn generate_pairs_for_round(&mut self, round_index: usize) -> Vec<Pair> {
        self.current_round_number = round_index;
        let num_items = self.id_map.len();

        let effective_strategy = get_effective_strategy(
            self.config.strategy,
            num_items,
            &self.games_played,
            self.current_round_number,
            self.config.min_games_before_strategy,
            self.config.number_of_rounds,
        );

        let index_pairs = match effective_strategy {
            Strategy::Balanced => generate_balanced_pairings_indexed(
                num_items,
                self.current_round_number,
                &self.current_ratings,
                self.config.matchmaking_sharpness,
            ),
            Strategy::TopHeavy => {
                let top_k_probs = self.mcmc_top_k_probs.as_ref()
                    .expect("TopHeavy strategy requires mcmc_top_k_probs to be set before generating pairs");
                let sample_means = self.mcmc_sample_means.as_ref()
                    .expect("TopHeavy strategy requires mcmc_sample_means to be set before generating pairs");

                generate_top_heavy_pairings_indexed(
                    num_items,
                    self.current_round_number,
                    top_k_probs,
                    sample_means,
                    self.config.matchmaking_sharpness,
                )
            }
        };

        // Convert index pairs to ID pairs
        index_pairs.into_iter().map(|(a, b)| {
            (self.id_map.to_id(a), self.id_map.to_id(b))
        }).collect()
    }

    /// Record comparison results from a round.
    pub fn record_results(&mut self, results: &[ComparisonInput]) {
        for result in results {
            self.completed_comparisons.push(*result);
            let idx1 = self.id_map.to_idx(result.item1);
            let idx2 = self.id_map.to_idx(result.item2);
            self.games_played[idx1] += 1;
            self.games_played[idx2] += 1;
        }
    }

    /// Update current rating estimates using Bradley-Terry MLE.
    pub fn update_current_ratings(&mut self) {
        if self.completed_comparisons.is_empty() {
            return;
        }

        let num_items = self.id_map.len();
        let indexed = self.id_map.convert_comparisons(&self.completed_comparisons);
        let mut bt = BradleyTerry::new(num_items, &indexed, 0.01);
        bt.calculate_scores(30);

        for i in 0..num_items {
            self.current_ratings[i] = bt.get_score(i);
        }
    }

    pub fn current_ratings(&self) -> &[f64] {
        &self.current_ratings
    }

    pub fn completed_comparison_count(&self) -> usize {
        self.completed_comparisons.len()
    }
}

/// Calculate pairs for a round using progressive scaling.
pub fn calculate_pairs_for_round(num_items: usize, round_number: usize) -> usize {
    let base_pairs = num_items / 2;

    if base_pairs >= CONCURRENCY_LIMIT {
        return base_pairs;
    }

    if base_pairs == 0 {
        return 0;
    }

    let min_multiplier_to_saturate = (CONCURRENCY_LIMIT + base_pairs - 1) / base_pairs;
    let effective_max_multiplier = min_multiplier_to_saturate.min(MAX_ROUND_MULTIPLIER);
    let multiplier = round_number.min(effective_max_multiplier);

    base_pairs * multiplier
}

/// Calculate total expected comparisons across all rounds.
pub fn calculate_total_expected_comparisons(num_items: usize, number_of_rounds: usize) -> usize {
    (1..=number_of_rounds).map(|r| calculate_pairs_for_round(num_items, r)).sum()
}

/// Calculate rounds needed to reach target comparisons.
pub fn calculate_rounds_for_target_comparisons(num_items: usize, target_comparisons: usize) -> usize {
    if num_items < 2 || target_comparisons == 0 {
        return 0;
    }

    let mut total = 0;
    let mut rounds = 0;
    while total < target_comparisons {
        rounds += 1;
        total += calculate_pairs_for_round(num_items, rounds);
    }
    rounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_pairs_for_round_large_list() {
        assert_eq!(calculate_pairs_for_round(1000, 1), 500);
        assert_eq!(calculate_pairs_for_round(1000, 5), 500);
    }

    #[test]
    fn test_calculate_pairs_for_round_small_list() {
        assert_eq!(calculate_pairs_for_round(10, 1), 5);
        assert_eq!(calculate_pairs_for_round(10, 2), 10);
        assert_eq!(calculate_pairs_for_round(10, 4), 20);
    }

    #[test]
    fn test_calculate_rounds_for_target() {
        let num_items = 100;
        let target = 500;
        let rounds = calculate_rounds_for_target_comparisons(num_items, target);
        let total = calculate_total_expected_comparisons(num_items, rounds);
        assert!(total >= target);

        if rounds > 1 {
            let total_minus_one = calculate_total_expected_comparisons(num_items, rounds - 1);
            assert!(total_minus_one < target);
        }
    }

    fn make_input(id1: i64, id2: i64, prob: f64) -> ComparisonInput {
        ComparisonInput { item1: id1, item2: id2, item1_win_probability: prob }
    }

    #[test]
    fn test_engine_basic_workflow() {
        let item_ids = vec![10, 20, 30, 40];
        let config = EngineConfig {
            strategy: Strategy::Balanced,
            matchmaking_sharpness: 1.0,
            min_games_before_strategy: 3,
            number_of_rounds: Some(5),
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
            strategy: Strategy::Balanced,
            matchmaking_sharpness: 1.0,
            min_games_before_strategy: 3,
            number_of_rounds: None,
        };
        let _ = RankingEngine::new(&[1], config);
    }

    #[test]
    #[should_panic(expected = "Duplicate item ID")]
    fn test_engine_rejects_duplicate_ids() {
        let config = EngineConfig {
            strategy: Strategy::Balanced,
            matchmaking_sharpness: 1.0,
            min_games_before_strategy: 3,
            number_of_rounds: None,
        };
        let _ = RankingEngine::new(&[1, 2, 1], config);
    }
}
