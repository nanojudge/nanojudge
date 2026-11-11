use std::collections::HashMap;

use crate::sha256::sha256;

fn truncated_sha256(input: &[u8]) -> u64 {
    let digest = sha256(input);
    u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3],
        digest[4], digest[5], digest[6], digest[7],
    ])
}

/// Stable identity hash for item text. Truncated SHA-256.
pub fn item_hash(text: &str) -> u64 {
    truncated_sha256(format!("item:{text}").as_bytes())
}

/// Stable identity hash for the full text of a prompt sent to a judge. Truncated SHA-256.
pub fn prompt_hash(text: &str) -> u64 {
    truncated_sha256(format!("prompt:{text}").as_bytes())
}

/// Stable identity hash for a judge (endpoint + model). Truncated SHA-256.
pub fn judge_hash(endpoint: &str, model: &str) -> u64 {
    truncated_sha256(format!("judge:{endpoint}\0{model}").as_bytes())
}

/// One edge consumed by the Bradley–Terry scoring model.
///
/// Items are identified by caller-provided `i64` IDs.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Edge {
    /// ID of first item.
    pub item1: i64,
    /// ID of second item.
    pub item2: i64,
    /// Verdict distribution from item1's perspective:
    /// `[P(item1 wins), P(item2 wins)]`.
    /// Sums to ~1. In text mode this is a one-hot vector; in logprobs mode it is the
    /// judge's full verdict distribution. Caller filters out failed edges first.
    pub category_probs: [f64; 2],
    /// Presentation slot item1 occupied in the prompt, and item2's slot. For a
    /// two-item judgement this is `(0, 1)` — item1 shown first, item2 second.
    /// For a three-item lineup judgement folded into edges, these are the
    /// A/B/C slots (0/1/2) the two items occupied. The scoring engine estimates a
    /// per-judge advantage for each slot and corrects for it, generalizing the
    /// pairwise first-position bias to any number of slots. The highest slot index
    /// present across all edges is the reference (advantage pinned to 0).
    pub slot1: u8,
    pub slot2: u8,
    /// Hash of endpoint+model identifying the judge that produced this edge.
    pub judge_id: u64,
    /// Likelihood weight for this edge. Two-item judgements use 1.0. Edges
    /// decomposed from a lineup judgement use `df / k` where `k` is the
    /// number of surviving edges and `df` is `size - 1`, one per ranked place
    /// but the last. See `lineup::lineup_verdict_to_edges` for details.
    pub weight: f64,
}

impl Edge {
    /// Build the edge produced by a two-item judgement.
    pub fn new(item1: i64, item2: i64, category_probs: [f64; 2], judge_id: u64) -> Self {
        Edge { item1, item2, category_probs, slot1: 0, slot2: 1, judge_id, weight: 1.0 }
    }
}

/// Information about the judge panel passed to the scoring engine.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JudgeInfo {
    /// The set of judge IDs present in this tournament.
    /// Order determines the internal index used for parameter arrays.
    pub judge_ids: Vec<u64>,
    /// Whether judges provide logprobs (full category distributions rather
    /// than one-hot verdicts).
    pub logprobs_mode: bool,
}

/// Per-judge analytics from Bradley-Terry estimation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JudgeAnalytics {
    pub judge_id: u64,
    /// Positional bias in probability space. 0.5 = no bias, >0.5 = favors first item.
    pub positional_bias: f64,
    pub positional_bias_ci: (f64, f64),
    /// Number of edges this judge contributed.
    pub num_edges: usize,
}

/// A ranked item with point estimate and confidence interval bounds.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RankedItem {
    /// Item ID.
    pub item: i64,
    /// MAP log-strength estimate.
    pub score: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Options for Laplace Bradley-Terry scoring via `run_scoring()`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringOptions {
    /// Confidence interval level (e.g. 0.95).
    pub confidence_level: f64,
    /// Top-heavy item-selection weighting. `None` skips it entirely (uniform
    /// pairing needs no selection weights, and the result's `selection_weights`
    /// is then `None`). `Some(sharpness)` computes, per item, the uncertainty
    /// ratio around the top-k boundary (see `selection_top_k`): with
    /// `A = P(item is in the top k)`, the ratio is
    /// `min(A, 1−A) / max(A, 1−A)` — 1 when the item is as likely in the top k
    /// as out of it, decaying toward 0 as the item resolves confidently onto
    /// either side — then raises that ratio to `sharpness`.
    /// Lower sharpness flattens the distribution (more exploration); `1.0`
    /// leaves the ratios as-is. Must be finite and > 0.
    pub selection_sharpness: Option<f64>,
    /// How many top places the selection aims to settle. Each item's selection
    /// weight comes from its chance of being among the `selection_top_k`
    /// strongest items when every item's strength is drawn from its posterior
    /// and the whole field is ranked. `1` aims at first place (classic
    /// top-heavy); `10` is right for "find the top ten, order within them
    /// doesn't matter", since edges then concentrate on the items that might
    /// or might not be in the top ten while items confidently in or out shed
    /// weight. Must be in `[1, num_items]`. Ignored when `selection_sharpness`
    /// is `None`.
    pub selection_top_k: usize,
    /// Minimum uncertainty ratio an item needs to stay a selection candidate.
    /// Items below are dropped to weight 0, except the two highest-ratio items,
    /// which are always kept so a pair can always be formed. In [0, 1); 0
    /// disables the cutoff. Ignored when `selection_sharpness` is `None`.
    pub selection_cutoff: f64,
    /// Proportional-fair coverage pull. Each item's ratio weight is divided by
    /// `edge_counts^selection_coverage`, pulling its cumulative edge count
    /// toward its ratio-implied share. 0 disables it (pure ratio sampling); 1 is
    /// standard proportional-fair; > 1 over-corrects toward equal coverage. Must
    /// be finite and >= 0. Ignored when `selection_sharpness` is `None`.
    pub selection_coverage: f64,
    /// How many random draws of the whole field estimate each item's chance of
    /// being in the top `selection_top_k`. More draws give a steadier estimate
    /// at proportionally more computation per scoring call. Must be >= 1.
    /// Ignored when `selection_sharpness` is `None`.
    pub selection_draws: usize,
    /// Seed for those random draws. Pass a different value on every scoring
    /// call so each refit sees fresh draws; the same value reproduces the same
    /// selection weights. Ignored when `selection_sharpness` is `None`.
    pub selection_seed: u64,
    /// Ghost player regularization strength (e.g. 0.01).
    pub regularization_strength: f64,
    /// Prior variance on log-strengths. Default: 10.0.
    pub prior_tau2: f64,
    /// Prior variance on positional bias (logit space). Default: 2.0.
    pub bias_prior_tau2: f64,
    /// Prior mean for positional bias in logit space. Default: 0.0 (= 0.5 probability = no bias).
    pub bias_prior_logit: f64,
}

/// Result from `run_scoring()`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringResult {
    /// Ranked items, sorted by score descending.
    pub rankings: Vec<RankedItem>,
    /// Top-heavy item-selection weights per item, in the same order as input
    /// `item_ids`: each item's sharpened uncertainty ratio around the top-k
    /// boundary, with sub-cutoff items zeroed. Drives top-heavy pairing: the
    /// first item of each pair is drawn from these weights. `None` when
    /// `selection_sharpness` was `None`.
    pub selection_weights: Option<Vec<f64>>,
    /// ln P(every item sits on its posterior-favored side of the top-k
    /// boundary): the sum over all items of `ln(max(A, 1−A))`, where `A` is
    /// the item's chance of being in the top `selection_top_k`, treating the
    /// per-item side probabilities as independent. In (−∞, 0];
    /// confidence-based early stopping fires when this reaches
    /// `ln(stop_confidence)`. Items far from the boundary contribute ~0, so
    /// the sum is dominated by the few contesting it. `None` when
    /// `selection_sharpness` was `None`.
    pub partition_log_confidence: Option<f64>,
    /// Per-item posterior mean log-strength, in the same order as the input
    /// `item_ids` (`rankings` is the score-sorted view of the same values).
    pub item_means: Vec<f64>,
    /// Per-item posterior std of log-strength, same order as input `item_ids`.
    /// Estimated from the inverse Hessian with deterministic matrix-free probes.
    /// Together with `item_means`, this feeds uncertainty-integrated matchmaking
    /// via `RankingEngine::set_current_posterior`.
    pub item_stds: Vec<f64>,
    /// Per-judge analytics. One entry per judge, in ascending `judge_id` order.
    pub judge_analytics: Vec<JudgeAnalytics>,
    /// Panel-level positional bias: edge-count-weighted average of the
    /// judges' estimated bias probabilities (equal weights when there are no
    /// edges).
    pub panel_positional_bias: f64,
    /// Approximate credible interval for `panel_positional_bias`, propagated from
    /// the per-judge Laplace variances with the delta method.
    pub panel_positional_bias_ci: (f64, f64),
}

/// A two-item lineup selected for judgement.
pub type Pair = (i64, i64);

/// An ordered lineup of items presented together in one judgement.
/// Holds between 2 and 9 caller IDs, in presentation order. Its judged
/// ranking is folded into edges by
/// [`crate::lineup::lineup_verdict_to_edges`].
pub type Lineup = Vec<i64>;

/// Internal indexed edge (usize indices, not caller IDs).
/// (item1_idx, item2_idx, win_prob, judge_internal_idx, slot1, slot2, weight)
///
/// `win_prob` is P(item1 wins), taken directly from `category_probs[0]`.
/// `slot1`/`slot2` are the
/// presentation slots the two items occupied (for per-slot bias correction).
/// `weight` scales this edge's contribution to the likelihood.
pub(crate) type IndexedEdge = (usize, usize, f64, usize, u8, u8, f64);

/// Internal indexed pair (usize indices, not caller IDs).
pub(crate) type IndexedPair = (usize, usize);

/// Internal indexed lineup (usize indices, not caller IDs).
pub(crate) type IndexedLineup = Vec<usize>;

/// Maps between caller-provided i64 IDs and internal 0..N indices.
pub(crate) struct IdMap {
    ids: Vec<i64>,
    id_to_idx: HashMap<i64, usize>,
}

impl IdMap {
    pub fn from_ids(ids: &[i64]) -> Self {
        let mut id_to_idx = HashMap::with_capacity(ids.len());
        for (idx, &id) in ids.iter().enumerate() {
            let prev = id_to_idx.insert(id, idx);
            assert!(prev.is_none(), "Duplicate item ID: {}", id);
        }
        IdMap {
            ids: ids.to_vec(),
            id_to_idx,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn to_idx(&self, id: i64) -> usize {
        *self.id_to_idx.get(&id)
            .unwrap_or_else(|| panic!("Unknown item ID: {}", id))
    }

    pub fn to_id(&self, idx: usize) -> i64 {
        self.ids[idx]
    }

    pub fn convert_edges(&self, edges: &[Edge], judge_id_to_idx: &HashMap<u64, usize>) -> Vec<IndexedEdge> {
        edges.iter().map(|c| {
            let judge_idx = *judge_id_to_idx.get(&c.judge_id)
                .unwrap_or_else(|| panic!("Unknown judge_id: {}", c.judge_id));
            let win_prob = c.category_probs[0];
            (self.to_idx(c.item1), self.to_idx(c.item2), win_prob, judge_idx, c.slot1, c.slot2, c.weight)
        }).collect()
    }

}
