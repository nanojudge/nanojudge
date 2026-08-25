//! nanojudge-core: Pure-computation ranking engine.
//!
//! Edges → Bradley-Terry scores → ranked list with confidence intervals.
//! No IO, no HTTP, no filesystem — just math. Bring your own LLM.
//!
//! Items are identified by caller-provided `i64` IDs. The crate handles the
//! internal mapping to efficient array indices — callers never think about indices.
//!
//! # Panics, not Results
//!
//! This crate treats invalid caller-supplied data as a programming error, not a
//! recoverable condition: precondition violations (duplicate IDs, unknown IDs,
//! mismatched lengths) panic immediately with a message naming the violation.
//! There is no error type to handle and no silent repair of bad input — validate
//! at your boundary, then trust the call. Each public function documents its
//! panic conditions in a `# Panics` section.
//!
//! # Quick start
//!
//! ```rust
//! use nanojudge_core::{run_scoring, Edge, JudgeInfo, ScoringOptions, judge_hash};
//!
//! let item_ids = vec![100, 200, 300]; // your IDs — any i64 values
//! let judge_id = judge_hash("http://localhost:8000", "my-model");
//!
//! // category_probs = [P(item1 wins), P(item2 wins)]
//! let edges = vec![
//!     Edge { slot1: 0, slot2: 1, item1: 100, item2: 200, category_probs: [1.0, 0.0], judge_id, weight: 1.0 },
//!     Edge { slot1: 0, slot2: 1, item1: 200, item2: 300, category_probs: [0.7, 0.3], judge_id, weight: 1.0 },
//! ];
//!
//! let judge_info = JudgeInfo {
//!     judge_ids: vec![judge_id],
//!     logprobs_mode: true,
//! };
//!
//! let result = run_scoring(&item_ids, &edges, &ScoringOptions {
//!     confidence_level: 0.95,
//!     selection_sharpness: None,
//!     anchor_index: 0.0,
//!     selection_cutoff: 0.0005,
//!     selection_coverage: 1.0,
//!     target_prior_edges: 10.0,
//!     regularization_strength: 0.01,
//!     prior_tau2: 10.0,
//!     bias_prior_tau2: 2.0,
//!     bias_prior_logit: 0.0,
//! }, &judge_info);
//!
//! for r in &result.rankings {
//!     println!("Item {}: {:.4} [{:.4}, {:.4}]", r.item, r.score, r.lower_bound, r.upper_bound);
//! }
//! ```

// Compile and run the README's code examples as doctests so they cannot
// drift from the real API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub mod bradley_terry;
pub mod constants;
pub mod engine;
pub mod laplace_bt;
pub mod pairing;
pub mod scoring;
pub mod seed;
mod sha256;
pub mod lineup;
pub mod types;

// Re-export primary public API at crate root.
pub use engine::{
    calculate_budget, judgements_needed_for_every_item_to_appear_once, EngineConfig,
    RankingEngine,
};
pub use pairing::{
    calculate_info_gain, calculate_integrated_info_gain,
    generate_uniform_pairings, generate_top_heavy_pairings,
    get_effective_judgement_distribution, JudgementDistribution,
};
pub use scoring::run_scoring;
pub use lineup::winner_dist_to_edges;
pub use types::{
    item_hash, judge_hash, Edge, JudgeAnalytics, JudgeInfo, Pair, RankedItem,
    ScoringOptions, ScoringResult,
};
