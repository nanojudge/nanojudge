/// nanojudge-core: Pure-computation ranking engine.
///
/// Pairwise comparison → Bradley-Terry scores → ranked list with confidence intervals.
/// No IO, no HTTP, no filesystem — just math. Bring your own LLM.
///
/// Items are identified by caller-provided `i64` IDs. The crate handles the
/// internal mapping to efficient array indices — callers never think about indices.
///
/// # Quick start
///
/// ```rust
/// use nanojudge_core::{run_scoring, ComparisonInput, ScoringOptions};
///
/// let item_ids = vec![100, 200, 300]; // your IDs — any i64 values
///
/// let comparisons = vec![
///     ComparisonInput { item1: 100, item2: 200, item1_win_probability: 0.8 },
///     ComparisonInput { item1: 200, item2: 300, item1_win_probability: 0.7 },
/// ];
///
/// let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
///     iterations: 200,
///     burn_in: 100,
///     confidence_level: 0.95,
///     top_k: 0,
///     warm_start: None,
///     regularization_strength: 0.01,
/// });
///
/// for r in &result.rankings {
///     println!("Item {}: {:.4} [{:.4}, {:.4}]", r.item, r.score, r.lower_bound, r.upper_bound);
/// }
/// ```

pub mod bradley_terry;
pub mod constants;
pub mod engine;
pub mod gaussian_bt;
pub mod pairing;
pub mod scoring;
pub mod types;

// Re-export primary public API at crate root.
pub use engine::{
    calculate_pairs_for_round, calculate_rounds_for_target_comparisons,
    calculate_total_expected_comparisons, EngineConfig, RankingEngine,
};
pub use pairing::{
    calculate_info_gain, generate_balanced_pairings, generate_top_heavy_pairings,
    get_effective_strategy, Strategy,
};
pub use scoring::run_scoring;
pub use types::{ComparisonInput, Pair, RankedItem, ScoringOptions, ScoringResult};
