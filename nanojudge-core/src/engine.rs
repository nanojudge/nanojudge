/// Ranking engine orchestrator.
///
/// Adapted for a pure computation crate — no async, no HTTP, no IO.
/// The caller obtains judgements externally, then feeds their edges back.
///
/// Items are identified by caller-provided `i64` IDs.
use crate::bradley_terry::BradleyTerry;
use crate::constants::INITIAL_BRADLEY_TERRY_RATING;
use crate::pairing::{
    calculate_lineups_for_round, generate_uniform_pairings_indexed,
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
    /// across rounds.
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
    /// `generate_pairs_for_round()` when using the TopHeavy distribution past
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

    /// Generate pairs for a round. Returns pairs of item IDs.
    ///
    /// # Panics
    ///
    /// Panics if the round's effective distribution is top-heavy and the
    /// posterior summary is missing or malformed. The effective distribution is top-heavy
    /// when the engine is configured with `JudgementDistribution::TopHeavy`
    /// and every item has reached `min_uniform_edges` (the uniform stage needs
    /// no posterior summary). In a top-heavy round `selection_weights` must be set, with
    /// one entry per item.
    pub fn generate_pairs_for_round(&mut self, _round_index: usize) -> Vec<Pair> {
        let pairs_count = calculate_pairs_for_round(self.id_map.len());
        self.generate_pairs(pairs_count)
    }

    /// The judgement distribution the next generated batch will use, given the
    /// edge counts so far. Top-heavy only once every item has reached
    /// `min_uniform_edges` (otherwise uniform pairing). Round subdivision into
    /// refit chunks derives from this via `round_chunk_sizes()`: only top-heavy
    /// batches are safe to split, since the uniform stage pairs every item
    /// exactly once per full round.
    pub fn effective_distribution(&self) -> JudgementDistribution {
        get_effective_judgement_distribution(
            self.config.judgement_distribution,
            self.id_map.len(),
            &self.edge_counts,
            self.config.min_uniform_edges,
        )
    }

    /// Decide how many refit chunks the next round splits into, returning the
    /// chunk sizes (each a `generate_pairs()` batch; they sum to one full round).
    ///
    /// `refits_per_round = 1` keeps the round whole. Higher values subdivide it
    /// into that many near-equal chunks so the caller can refit scoring and
    /// re-derive selection weights between chunks — but only when the round's
    /// effective distribution is top-heavy. Uniform rounds pair every item
    /// exactly once, so they are never subdivided and always come back as a
    /// single full-round chunk. Encoding that rule here means callers cannot
    /// get the policy wrong.
    ///
    /// # Panics
    ///
    /// Panics if `refits_per_round` is 0.
    pub fn round_chunk_sizes(&self, refits_per_round: usize) -> Vec<usize> {
        assert!(refits_per_round >= 1, "refits_per_round must be at least 1");
        let pairs_per_round = calculate_pairs_for_round(self.id_map.len());
        if refits_per_round > 1
            && self.effective_distribution() == JudgementDistribution::TopHeavy
        {
            split_round_into_chunks(pairs_per_round, refits_per_round)
        } else {
            vec![pairs_per_round]
        }
    }

    /// The three-item lineup analogue of `round_chunk_sizes`: chunk sizes in lineups that
    /// sum to one full round (`calculate_lineups_for_round(num_items)`). Same
    /// policy — only top-heavy rounds subdivide (for mid-round refits); the
    /// uniform stage always returns a single full-round chunk.
    ///
    /// # Panics
    ///
    /// Panics if `refits_per_round` is 0.
    pub fn round_chunk_sizes_lineups(&self, refits_per_round: usize) -> Vec<usize> {
        assert!(refits_per_round >= 1, "refits_per_round must be at least 1");
        let lineups_per_round = calculate_lineups_for_round(self.id_map.len());
        if refits_per_round > 1
            && self.effective_distribution() == JudgementDistribution::TopHeavy
        {
            split_round_into_chunks(lineups_per_round, refits_per_round)
        } else {
            vec![lineups_per_round]
        }
    }

    /// Generate a batch of `pairs_count` pairs using the current effective
    /// distribution. A full round is `calculate_pairs_for_round(num_items)`
    /// pairs; passing a smaller count yields a sub-round batch (used to refit
    /// more often than once per round in top-heavy mode).
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

    /// Generate a batch of `lineups_count` three-item lineups using the
    /// current effective distribution. The three-item lineup analogue of `generate_pairs`: a
    /// full round is `calculate_lineups_for_round(num_items)` lineups. Each lineup
    /// receives one judgement, then is folded into edges by the caller via
    /// `lineup::winner_dist_to_edges` before being fed
    /// back through `record_edges`.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as `generate_pairs`: a top-heavy round
    /// with `selection_weights` unset or malformed.
    pub fn generate_lineups(&mut self, lineups_count: usize) -> Vec<Lineup> {
        let num_items = self.id_map.len();

        let effective_judgement_distribution = self.effective_distribution();

        let index_lineups = match effective_judgement_distribution {
            JudgementDistribution::Uniform => generate_uniform_lineups_indexed(
                num_items,
                lineups_count,
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
                    selection_weights,
                    &self.current_ratings,
                    self.current_stds.as_deref(),
                    self.config.matchmaking_sharpness,
                    &mut self.rng,
                )
            }
        };

        // Convert index lineups to ID lineups.
        index_lineups.into_iter().map(|(a, b, c)| {
            (self.id_map.to_id(a), self.id_map.to_id(b), self.id_map.to_id(c))
        }).collect()
    }

    /// Record edges from a round.
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
        let mut bt = BradleyTerry::new(num_items, &indexed, 0.01);
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

/// Calculate pairs for a single round: every item gets compared once.
pub fn calculate_pairs_for_round(num_items: usize) -> usize {
    num_items / 2
}

/// Calculate total expected two-item judgements across all rounds.
pub fn calculate_total_expected_judgements(num_items: usize, number_of_rounds: usize) -> usize {
    calculate_pairs_for_round(num_items) * number_of_rounds
}

/// Split a round's `total` pairs into `parts` chunk sizes that sum to `total`,
/// as evenly as possible (the first `total % parts` chunks get one extra).
/// Zero-sized chunks (when `parts` exceeds `total`) are dropped, so the result
/// always has between 1 and `min(parts, total)` entries.
pub fn split_round_into_chunks(total: usize, parts: usize) -> Vec<usize> {
    if parts <= 1 || total == 0 {
        return vec![total];
    }
    let base = total / parts;
    let remainder = total % parts;
    (0..parts)
        .map(|i| base + usize::from(i < remainder))
        .filter(|&size| size > 0)
        .collect()
}

/// Calculate rounds needed to reach a target number of two-item judgements.
pub fn calculate_rounds_for_target_judgements(num_items: usize, target_judgements: usize) -> usize {
    let pairs_per_round = calculate_pairs_for_round(num_items);
    if pairs_per_round == 0 || target_judgements == 0 {
        return 0;
    }
    target_judgements.div_ceil(pairs_per_round)
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
    fn test_calculate_total_expected_judgements() {
        assert_eq!(calculate_total_expected_judgements(10, 5), 25);
        assert_eq!(calculate_total_expected_judgements(78, 10), 390);
        assert_eq!(calculate_total_expected_judgements(1000, 3), 1500);
    }

    #[test]
    fn test_calculate_rounds_for_target() {
        assert_eq!(calculate_rounds_for_target_judgements(100, 500), 10);
        assert_eq!(calculate_rounds_for_target_judgements(100, 501), 11);
        assert_eq!(calculate_rounds_for_target_judgements(100, 0), 0);
        assert_eq!(calculate_rounds_for_target_judgements(1, 100), 0);
    }

    #[test]
    fn test_split_round_into_chunks_even() {
        assert_eq!(split_round_into_chunks(200, 4), vec![50, 50, 50, 50]);
    }

    #[test]
    fn test_split_round_into_chunks_uneven_sums_to_total() {
        let chunks = split_round_into_chunks(201, 4);
        assert_eq!(chunks, vec![51, 50, 50, 50]);
        assert_eq!(chunks.iter().sum::<usize>(), 201);
    }

    #[test]
    fn test_split_round_into_chunks_single_part() {
        assert_eq!(split_round_into_chunks(200, 1), vec![200]);
    }

    #[test]
    fn test_split_round_into_chunks_more_parts_than_pairs() {
        // 3 pairs, 8 requested chunks → three 1-pair chunks, zeros dropped.
        assert_eq!(split_round_into_chunks(3, 8), vec![1, 1, 1]);
    }

    fn top_heavy_engine(num_items: usize) -> RankingEngine {
        let item_ids: Vec<i64> = (0..num_items as i64).collect();
        RankingEngine::new(&item_ids, EngineConfig {
            judgement_distribution: JudgementDistribution::TopHeavy,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            seed: Some(1),
        })
    }

    #[test]
    fn test_round_chunk_sizes_uniform_stage_never_subdivides() {
        // Fresh engine: nobody has reached min_uniform_edges, so the effective
        // distribution is uniform and the round stays whole despite refits > 1.
        let engine = top_heavy_engine(10);
        assert_eq!(engine.round_chunk_sizes(4), vec![5]);
    }

    #[test]
    fn test_round_chunk_sizes_top_heavy_subdivides() {
        let mut engine = top_heavy_engine(10);
        engine.edge_counts = vec![3; 10]; // uniform stage complete
        assert_eq!(engine.round_chunk_sizes(4), vec![2, 1, 1, 1]);
        assert_eq!(engine.round_chunk_sizes(1), vec![5]);
    }

    #[test]
    fn test_round_chunk_sizes_uniform_config_never_subdivides() {
        let item_ids: Vec<i64> = (0..10).collect();
        let mut engine = RankingEngine::new(&item_ids, EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            seed: Some(1),
        });
        engine.edge_counts = vec![10; 10];
        assert_eq!(engine.round_chunk_sizes(4), vec![5]);
    }

    #[test]
    #[should_panic(expected = "refits_per_round must be at least 1")]
    fn test_round_chunk_sizes_zero_refits_panics() {
        top_heavy_engine(10).round_chunk_sizes(0);
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

        let results: Vec<Edge> = pairs.iter()
            .map(|(a, b)| make_input(*a, *b, 0.7))
            .collect();

        engine.record_edges(&results);
        engine.update_current_ratings();

        assert_eq!(engine.completed_edge_count(), pairs.len());
    }

    #[test]
    fn test_set_current_posterior_and_mle_refit_clears_stds() {
        let item_ids: Vec<i64> = vec![10, 20, 30];
        let config = EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
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
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 3,
            seed: None,
        };

        let mut engine = RankingEngine::new(&item_ids, config);

        for round in 0..342 {
            let pairs = engine.generate_pairs_for_round(round);
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
            seed: None,
        };
        let _ = RankingEngine::new(&[1, 2, 1], config);
    }
}
