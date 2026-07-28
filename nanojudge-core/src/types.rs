use std::collections::HashMap;

/// Deterministic FNV-1a hash. Same input → same u64, always.
///
/// Rust's `DefaultHasher` (SipHash) is randomized per process to prevent HashDoS,
/// which means it produces different hashes across runs. We need stable hashes for
/// judge identity in saved comparisons, so we use FNV-1a instead.
pub fn stable_hash(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Input format for a single pairwise comparison.
///
/// Items are identified by caller-provided `i64` IDs.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonInput {
    /// ID of first item.
    pub item1: i64,
    /// ID of second item.
    pub item2: i64,
    /// Verdict distribution from item1's perspective:
    /// `[P(item1 wins), P(item2 wins)]`.
    /// Sums to ~1. In text mode this is a one-hot vector; in logprobs mode it is the
    /// judge's full verdict distribution. Caller filters out failed comparisons first.
    pub category_probs: [f64; 2],
    /// Presentation slot item1 occupied in the prompt, and item2's slot. For a
    /// plain pairwise comparison this is `(0, 1)` — item1 shown first, item2
    /// second. For a 3-way comparison folded into pairwise edges, these are the
    /// A/B/C slots (0/1/2) the two items occupied. The scoring engine estimates a
    /// per-judge advantage for each slot and corrects for it, generalizing the
    /// pairwise first-position bias to any number of slots. The highest slot index
    /// present across all comparisons is the reference (advantage pinned to 0).
    pub slot1: u8,
    pub slot2: u8,
    /// Hash of endpoint+model identifying which judge produced this comparison.
    pub judge_id: u64,
}

impl ComparisonInput {
    /// Build a plain pairwise comparison: item1 in slot 0 (shown first), item2 in
    /// slot 1 (shown second).
    pub fn pairwise(item1: i64, item2: i64, category_probs: [f64; 2], judge_id: u64) -> Self {
        ComparisonInput { item1, item2, category_probs, slot1: 0, slot2: 1, judge_id }
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
    /// Number of comparisons this judge contributed.
    pub num_comparisons: usize,
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
    /// ratio around the anchor (see `anchor_index`): with
    /// `A = P(item beats the anchor)` under independent Gaussian summaries of
    /// both posteriors — `Φ((mean − anchor_mean) / sqrt(std² + anchor_std²))`,
    /// so a still-uncertain anchor widens every split — the ratio is
    /// `min(A, 1−A) / max(A, 1−A)` — 1 when the item's
    /// posterior straddles the anchor, decaying toward 0 as the item resolves
    /// confidently above or below it — then raises that ratio to `sharpness`.
    /// Lower sharpness flattens the distribution (more exploration); `1.0`
    /// leaves the ratios as-is. Must be finite and > 0.
    pub selection_sharpness: Option<f64>,
    /// Which rank anchors the selection target, as a 0-based (possibly
    /// fractional) index into the items sorted by posterior mean descending.
    /// `0.0` anchors on the top item (classic top-heavy); `9.0` anchors on the
    /// 10th-best — right for "find the top ten, order within them doesn't
    /// matter", since comparisons then concentrate on the boundary around rank
    /// 10 while items confidently inside or outside the top ten shed weight.
    /// Fractional values interpolate linearly (in log-strength space) between
    /// the two adjacent ranks: `0.5` targets the midpoint of the 1st and 2nd
    /// items' means. Must be finite, >= 0, and <= num_items − 1. Ignored when
    /// `selection_sharpness` is `None`.
    pub anchor_index: f64,
    /// Minimum uncertainty ratio an item needs to stay a selection candidate.
    /// Items below are dropped to weight 0, except the two highest-ratio items,
    /// which are always kept so a pair can always be formed. In [0, 1); 0
    /// disables the cutoff. Ignored when `selection_sharpness` is `None`.
    pub selection_cutoff: f64,
    /// Proportional-fair coverage pull. Each item's ratio weight is divided by
    /// `games_played^selection_coverage`, pulling its cumulative comparison count
    /// toward its ratio-implied share. 0 disables it (pure ratio sampling); 1 is
    /// standard proportional-fair; > 1 over-corrects toward equal coverage. Must
    /// be finite and >= 0. Ignored when `selection_sharpness` is `None`.
    pub selection_coverage: f64,
    /// Info-driven blend for the selection "target" (the reference strength each
    /// item's area is measured against). Early on the observed anchor strength is
    /// a poor estimate — the anchor item has played few games — so the target is
    /// blended with a prior-predicted anchor: the expected `anchor_index`-th
    /// order statistic of `num_items` draws from the strength prior. The prior
    /// counts as `target_prior_games` pseudo-games against the anchor's actual
    /// game count `g` (interpolated for fractional `anchor_index`):
    ///   `target = (g·observed_anchor + K·predicted_anchor) / (g + K)`.
    /// So the prediction dominates while the anchor has few games and fades as it
    /// plays more. `target_prior_games = 0` disables the blend (pure observed
    /// anchor). Must be finite and >= 0. Ignored when `selection_sharpness` is `None`.
    pub target_prior_games: f64,
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
    /// `item_ids`: each item's sharpened uncertainty ratio around the anchor
    /// (integrated over the anchor's own posterior uncertainty), with
    /// sub-cutoff items zeroed. Drives top-heavy pairing: the first
    /// item of each pair is drawn from these weights. `None` when
    /// `selection_sharpness` was `None`.
    pub selection_weights: Option<Vec<f64>>,
    /// ln P(every checked item sits on its posterior-favored side of the
    /// anchor): the sum over non-exempt items of `ln(max(A, 1−A))`, treating
    /// the per-item side probabilities as independent Gaussian marginals. In
    /// (−∞, 0]; confidence-based early stopping fires when this reaches
    /// `ln(stop_confidence)`. Items far from the anchor contribute ~0, so the
    /// sum is dominated by the boundary stragglers. An integer `anchor_index`
    /// exempts the anchor item itself — its side probability is ~0.5 by
    /// construction; a fractional anchor exempts nobody, since the boundary
    /// sits between two items that both must resolve against it. Measured
    /// against the observed anchor, not the `target_prior_games`-blended
    /// target that steers early exploration (items resolved against that
    /// inflated early target are not resolved against the actual anchor).
    /// 0.0 only for a single item. `None` when `selection_sharpness` was
    /// `None`.
    pub partition_log_confidence: Option<f64>,
    /// Per-item posterior mean log-strength, in the same order as the input
    /// `item_ids` (`rankings` is the score-sorted view of the same values).
    pub item_means: Vec<f64>,
    /// Per-item posterior std of log-strength, same order as input `item_ids`.
    /// Estimated from the inverse Hessian with deterministic matrix-free probes.
    /// Together with `item_means`, this feeds uncertainty-integrated matchmaking
    /// via `RankingEngine::set_current_posterior`.
    pub item_stds: Vec<f64>,
    /// Per-judge analytics. One entry per judge, same order as `JudgeInfo.judge_ids`.
    pub judge_analytics: Vec<JudgeAnalytics>,
    /// Panel-level positional bias: comparison-count-weighted average of the
    /// judges' estimated bias probabilities (equal weights when there are no
    /// comparisons).
    pub panel_positional_bias: f64,
    /// Approximate credible interval for `panel_positional_bias`, propagated from
    /// the per-judge Laplace variances with the delta method.
    pub panel_positional_bias_ci: (f64, f64),
}

/// A pairing: two item IDs to be compared.
pub type Pair = (i64, i64);

/// A triple: three item IDs to be compared together in one 3-way judgment.
/// The judge ranks all three; the winner-distribution over them is folded into
/// pairwise edges by [`crate::three_way::winner_dist_to_edges`].
pub type Triple = (i64, i64, i64);

/// Internal indexed comparison (usize indices, not caller IDs).
/// (item1_idx, item2_idx, win_prob, judge_internal_idx, slot1, slot2)
///
/// `win_prob` is P(item1 wins), collapsed from the 4-category distribution:
/// `probs[0] + probs[1]` (clear win + narrow win). `slot1`/`slot2` are the
/// presentation slots the two items occupied (for per-slot bias correction).
pub(crate) type IndexedComparison = (usize, usize, f64, usize, u8, u8);

/// Internal indexed pair (usize indices, not caller IDs).
pub(crate) type IndexedPair = (usize, usize);

/// Internal indexed triple (usize indices, not caller IDs).
pub(crate) type IndexedTriple = (usize, usize, usize);

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

    pub fn convert_comparisons(&self, comparisons: &[ComparisonInput], judge_id_to_idx: &HashMap<u64, usize>) -> Vec<IndexedComparison> {
        comparisons.iter().map(|c| {
            let judge_idx = *judge_id_to_idx.get(&c.judge_id)
                .unwrap_or_else(|| panic!("Unknown judge_id: {}", c.judge_id));
            let win_prob = c.category_probs[0];
            (self.to_idx(c.item1), self.to_idx(c.item2), win_prob, judge_idx, c.slot1, c.slot2)
        }).collect()
    }
}
