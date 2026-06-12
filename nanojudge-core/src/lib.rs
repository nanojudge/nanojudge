/// nanojudge-core: Pure-computation ranking engine.
///
/// Pairwise comparison → Bradley-Terry scores → ranked list with confidence intervals.
/// No IO, no HTTP, no filesystem — just math. Bring your own LLM.
///
/// Items are identified by caller-provided `i64` IDs. The crate handles the
/// internal mapping to efficient array indices — callers never think about indices.
///
/// # Panics, not Results
///
/// This crate treats invalid caller-supplied data as a programming error, not a
/// recoverable condition: precondition violations (duplicate IDs, unknown IDs,
/// mismatched lengths) panic immediately with a message naming the violation.
/// There is no error type to handle and no silent repair of bad input — validate
/// at your boundary, then trust the call. Each public function documents its
/// panic conditions in a `# Panics` section.
///
/// # Quick start
///
/// ```rust
/// use nanojudge_core::{run_scoring, ComparisonInput, JudgeInfo, ScoringOptions, stable_hash};
///
/// let item_ids = vec![100, 200, 300]; // your IDs — any i64 values
/// let judge_id = stable_hash("http://localhost:8000\0my-model");
///
/// // category_probs = [P(A clear win), P(B narrow win), P(C draw), P(D narrow loss), P(E clear loss)]
/// let comparisons = vec![
///     ComparisonInput { item1: 100, item2: 200, category_probs: [0.0, 1.0, 0.0, 0.0, 0.0], judge_id },
///     ComparisonInput { item1: 200, item2: 300, category_probs: [0.7, 0.3, 0.0, 0.0, 0.0], judge_id },
/// ];
///
/// let judge_info = JudgeInfo {
///     judge_ids: vec![judge_id],
///     logprobs_mode: true,
/// };
///
/// let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
///     iterations: 200,
///     burn_in: 100,
///     confidence_level: 0.95,
///     top_k: 0,
///     warm_start: None,
///     regularization_strength: 0.01,
///     prior_tau2: 10.0,
///     proposal_std: 0.3,
///     bias_prior_tau2: 2.0,
///     bias_proposal_std: 0.15,
///     bias_prior_logit: 0.0,
/// }, &judge_info);
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
    calculate_info_gain, generate_uniform_pairings, generate_top_heavy_pairings,
    get_effective_comparison_distribution, ComparisonDistribution,
};
pub use scoring::run_scoring;
pub use types::{
    stable_hash, ComparisonInput, JudgeAnalytics, JudgeInfo, Pair, RankedItem, ScoringOptions,
    ScoringResult, WarmStartState,
};
