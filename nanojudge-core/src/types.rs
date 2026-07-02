use std::collections::HashMap;

/// Deterministic FNV-1a hash. Same input → same u64, always.
///
/// Rust's `DefaultHasher` (SipHash) is randomized per process to prevent HashDoS,
/// which means it produces different hashes across runs. We need stable hashes for
/// judge identity (warm start state, saved comparisons), so we use FNV-1a instead.
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
    /// Categorical verdict distribution from item1's perspective:
    /// `[P(A clear win), P(B narrow win), P(C narrow loss), P(D clear loss)]`.
    /// Sums to ~1. In text mode this is a one-hot vector; in logprobs mode it is the
    /// judge's full per-bucket distribution. Caller filters out failed comparisons first.
    pub category_probs: [f64; 4],
    /// Hash of endpoint+model identifying which judge produced this comparison.
    pub judge_id: u64,
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

/// Per-judge analytics from MCMC estimation.
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

/// Warm start state for multi-judge MCMC.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WarmStartState {
    /// Item log-strengths (θ, the engines' native state), same order as item_ids.
    pub item_log_strengths: Vec<f64>,
    /// Per-judge positional bias in logit space, keyed by judge_id hash.
    /// Positive values mean the judge favors the first-listed item.
    pub judge_biases: Vec<(u64, f64)>,
}

/// A ranked item with point estimate and confidence interval bounds.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RankedItem {
    /// Item ID.
    pub item: i64,
    /// Point estimate (mean of posterior samples, or MLE score).
    pub score: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Which inference engine `run_scoring()` uses to turn comparisons into a
/// posterior summary `(mean, std)` per item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InferenceMode {
    /// Gaussian-BT MCMC sampler. Samples the full posterior.
    Mcmc,
    /// Linear-time Laplace approximation (default): finds the posterior mode by
    /// Newton-CG (no dense matrix) with a diagonal-Fisher std. Deterministic, no
    /// sampling, O(#comparisons); scales to large item counts.
    LaplaceLinear,
}

/// Options for `run_scoring()` — the unified scoring wrapper.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringOptions {
    /// Number of post-burn-in MCMC iterations (e.g. 5000 interim, 5000 final).
    pub iterations: usize,
    /// Burn-in iterations run before collecting samples. Applied even when `warm_start` is
    /// provided, so set to 0 (or a small value) if the warm start is already near the
    /// stationary distribution.
    pub burn_in: usize,
    /// Confidence interval level (e.g. 0.95).
    pub confidence_level: f64,
    /// Top-heavy item-selection weighting. `None` skips it entirely (uniform
    /// pairing needs no selection weights, and the result's `selection_weights`
    /// is then `None`). `Some(sharpness)` computes, per item, the area of its
    /// posterior lying above the top item's mean — P(item beats the current
    /// leader) under a Gaussian summary `(mean, std)` of its samples — then
    /// raises that area to `sharpness`. Lower sharpness flattens the distribution
    /// (more exploration); `1.0` leaves the areas as-is. Must be finite and > 0.
    pub selection_sharpness: Option<f64>,
    /// Minimum area (P(item beats the leader's mean)) an item needs to stay a
    /// selection candidate. Items below are dropped to weight 0, except the two
    /// highest-area items, which are always kept so a pair can always be formed.
    /// In [0, 1); 0 disables the cutoff. Ignored when `selection_sharpness` is
    /// `None`.
    pub selection_cutoff: f64,
    /// Proportional-fair coverage pull. Each item's area weight is divided by
    /// `games_played^selection_coverage`, pulling its cumulative comparison count
    /// toward its area-implied share. 0 disables it (pure area sampling); 1 is
    /// standard proportional-fair; > 1 over-corrects toward equal coverage. Must
    /// be finite and >= 0. Ignored when `selection_sharpness` is `None`.
    pub selection_coverage: f64,
    /// Info-driven blend for the selection "target" (the reference strength each
    /// item's area is measured against). Early on the observed top strength is a
    /// poor estimate — the leader has played few games — so the target is blended
    /// with a prior-predicted top, `E[max]` of `num_items` draws from the strength
    /// prior. The prior counts as `target_prior_games` pseudo-games against the
    /// leader's actual game count `g`:
    ///   `target = (g·observed_top + K·predicted_top) / (g + K)`.
    /// So the prediction dominates while the leader has few games and fades as it
    /// plays more. `target_prior_games = 0` disables the blend (pure observed
    /// top). Must be finite and >= 0. Ignored when `selection_sharpness` is `None`.
    pub target_prior_games: f64,
    /// Previous warm start state. `None` = cold start.
    pub warm_start: Option<WarmStartState>,
    /// Ghost player regularization strength (e.g. 0.01).
    pub regularization_strength: f64,
    /// Prior variance on log-strengths. Default: 10.0.
    pub prior_tau2: f64,
    /// MH proposal step size for strengths. Default: 0.3.
    pub proposal_std: f64,
    /// Prior variance on positional bias (logit space). Default: 2.0.
    pub bias_prior_tau2: f64,
    /// MH proposal step size for bias. Default: 0.15.
    pub bias_proposal_std: f64,
    /// Prior mean for positional bias in logit space. Default: 0.0 (= 0.5 probability = no bias).
    pub bias_prior_logit: f64,
    /// RNG seed for reproducible MCMC sampling. `None` = OS entropy.
    pub seed: Option<u64>,
    /// Which inference engine to use (MCMC sampling or Laplace approximation).
    pub inference: InferenceMode,
}

/// Result from `run_scoring()`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringResult {
    /// Ranked items, sorted by score descending.
    pub rankings: Vec<RankedItem>,
    /// Top-heavy item-selection weights per item, in the same order as input
    /// `item_ids`: each item's softened P(beat the leader's mean), with
    /// sub-cutoff items zeroed. Drives both-items-from-the-weights pairing.
    /// `None` when `selection_sharpness` was `None`.
    pub selection_weights: Option<Vec<f64>>,
    /// Warm start state for next round.
    pub warm_start_state: WarmStartState,
    /// Number of post-burn-in samples (for DB storage).
    pub sample_size: usize,
    /// Per-judge analytics. One entry per judge, same order as `JudgeInfo.judge_ids`.
    pub judge_analytics: Vec<JudgeAnalytics>,
    /// Panel-level positional bias: posterior mean of the comparison-count-weighted
    /// average of the judges' bias probabilities (equal weights when there are no
    /// comparisons).
    pub panel_positional_bias: f64,
    /// Credible interval for `panel_positional_bias`, computed from per-iteration
    /// posterior samples of the weighted average — not by averaging per-judge
    /// interval endpoints, which would be too wide. With a single judge this
    /// equals that judge's `positional_bias_ci`.
    pub panel_positional_bias_ci: (f64, f64),
}

/// A pairing: two item IDs to be compared.
pub type Pair = (i64, i64);

/// Internal indexed comparison (usize indices, not caller IDs).
/// (item1_idx, item2_idx, win_prob, judge_internal_idx)
///
/// `win_prob` is P(item1 wins), collapsed from the 4-category distribution:
/// `probs[0] + probs[1]` (clear win + narrow win).
pub(crate) type IndexedComparison = (usize, usize, f64, usize);

/// Internal indexed pair (usize indices, not caller IDs).
pub(crate) type IndexedPair = (usize, usize);

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
            let p = c.category_probs;
            let win_prob = p[0] + p[1];
            (self.to_idx(c.item1), self.to_idx(c.item2), win_prob, judge_idx)
        }).collect()
    }
}
