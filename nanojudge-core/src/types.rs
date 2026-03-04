use std::collections::HashMap;

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
    /// P(item1 wins) from logprob extraction, 0.0 to 1.0.
    /// Caller is responsible for filtering out failed comparisons before passing data in.
    pub item1_win_probability: f64,
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

/// Options for `run_scoring()` — the unified MCMC scoring wrapper.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringOptions {
    /// Number of post-burn-in MCMC iterations (e.g. 200 interim, 2000 final).
    pub iterations: usize,
    /// Burn-in iterations. Ignored when `warm_start` is provided.
    pub burn_in: usize,
    /// Confidence interval level (e.g. 0.95).
    pub confidence_level: f64,
    /// Compute P(top K) probabilities. 0 = skip.
    pub top_k: usize,
    /// Previous lambda state for warm-starting. `None` = cold start.
    /// Must be in the same order as `item_ids` passed to `run_scoring()`.
    pub warm_start: Option<Vec<f64>>,
    /// Ghost player regularization strength (e.g. 0.01).
    pub regularization_strength: f64,
}

/// Result from `run_scoring()`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringResult {
    /// Ranked items, sorted by score descending.
    pub rankings: Vec<RankedItem>,
    /// P(top K) probabilities per item, in the same order as input `item_ids`. `None` if `top_k == 0`.
    pub top_k_probs: Option<Vec<f64>>,
    /// Mean scores per item, in the same order as input `item_ids`. `None` if `top_k == 0`.
    pub sample_means: Option<Vec<f64>>,
    /// Lambda state for warm-starting next round, in the same order as input `item_ids`.
    pub state: Vec<f64>,
    /// Number of post-burn-in samples (for DB storage).
    pub sample_size: usize,
    /// Estimated positional bias in probability space (0.5 = no bias, >0.5 = item1 favored).
    pub positional_bias: f64,
    /// Confidence interval for positional bias in probability space.
    pub positional_bias_confidence_interval: (f64, f64),
}

/// A pairing: two item IDs to be compared.
pub type Pair = (i64, i64);

/// Internal indexed comparison (usize indices, not caller IDs).
pub(crate) type IndexedComparison = (usize, usize, f64);

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

    pub fn convert_comparisons(&self, comparisons: &[ComparisonInput]) -> Vec<IndexedComparison> {
        comparisons.iter().map(|c| {
            (self.to_idx(c.item1), self.to_idx(c.item2), c.item1_win_probability)
        }).collect()
    }
}
