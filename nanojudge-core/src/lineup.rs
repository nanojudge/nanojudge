//! Lineup judgement → edges.
//!
//! A judgement over a `k`-item lineup (`2 ≤ k ≤ 9`) is summarized as a
//! **winner-distribution**: the probability each of the `k` items is the best of
//! the lineup (`q_A`, `q_B`, ...). Under a Luce model this distribution is the
//! top-1 marginal, with `q_i ∝ s_i` (the items' latent strengths).
//!
//! The ranking engine consumes edges. This module converts one winner distribution
//! into up to `k · (k − 1) / 2` `Edge` values — one per unordered pair — via
//! the Luce ratio `P(i beats j) = q_i / (q_i + q_j)`. The engine then treats them
//! as ordinary, mutually consistent edges; the core likelihood is untouched.
//!
//! Because the ratio only depends on `q_i / q_j`, the winner-distribution need not
//! be exactly normalized — any positive scaling gives the same edges.
//!
//! # Edge weighting
//!
//! The degrees of freedom depend on the verdict's provenance, not on how many edges
//! survive. Logprobs mode carries `k − 1` df: the parser reads the first `k − 1`
//! ranking slots, each contributing one free parameter, and the last item's
//! probability is whatever residual remains. Text mode carries 1 df regardless of
//! `k` (a single categorical "who won"). Each surviving edge gets `df / m` where
//! `m` is the number of surviving edges:
//!
//! | Mode     | k | Edges | df | Weight each | Total |
//! |----------|---|-------|----|-------------|-------|
//! | Logprobs | 2 | 1     | 1  | 1           | 1     |
//! | Logprobs | 3 | 3     | 2  | 2/3         | 2     |
//! | Logprobs | 3 | 2     | 2  | 1           | 2     |
//! | Logprobs | 9 | 36    | 8  | 2/9         | 8     |
//! | Text     | 3 | 2     | 1  | 1/2         | 1     |
//! | Text     | 9 | 8     | 1  | 1/8         | 1     |

use crate::constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE};
use crate::types::Edge;

/// Convert a lineup's winner distribution into edges.
///
/// `item_ids` are the caller IDs in presentation order; `winner_probs[k]` is the
/// probability that `item_ids[k]` is the best of the lineup. Produces up to one
/// `Edge` per unordered pair, in index order (`(0,1)`, `(0,2)`, ..., `(1,2)`, ...),
/// each carrying `P(item1 beats item2)` from the Luce ratio, all attributed to
/// `judge_id`.
///
/// An edge is **dropped** when both its items have zero winner-probability: the Luce
/// ratio is then `0/0`, undefined, carrying no information.
///
/// `logprobs_mode` controls the degrees of freedom: `true` means the verdict came
/// from a full probability distribution (`k − 1` df even when an edge is dropped
/// because a probability rounded to zero), `false` means a text-mode winner-only
/// verdict (1 df).
///
/// # Panics
///
/// Panics if `item_ids` and `winner_probs` differ in length, if the lineup size is
/// outside `2..=9`, if any `winner_probs` entry is not finite or is negative, or if
/// the `item_ids` are not distinct.
pub fn winner_dist_to_edges(
    item_ids: &[i64],
    winner_probs: &[f64],
    judge_id: u64,
    logprobs_mode: bool,
) -> Vec<Edge> {
    let size = item_ids.len();
    assert_eq!(
        size,
        winner_probs.len(),
        "item_ids ({}) and winner_probs ({}) must have the same length",
        size,
        winner_probs.len()
    );
    assert!(
        (MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE).contains(&size),
        "lineup size must be between {MIN_LINEUP_SIZE} and {MAX_LINEUP_SIZE}, got {size}"
    );
    for &q in winner_probs {
        assert!(
            q.is_finite() && q >= 0.0,
            "winner_probs entries must be finite and non-negative, got {q}"
        );
    }
    for a in 0..size {
        for b in (a + 1)..size {
            assert!(
                item_ids[a] != item_ids[b],
                "lineup item_ids must be distinct, got {item_ids:?}"
            );
        }
    }

    let mut edges = Vec::with_capacity(size * (size - 1) / 2);
    for a in 0..size {
        for b in (a + 1)..size {
            let (qa, qb) = (winner_probs[a], winner_probs[b]);
            let total = qa + qb;
            if total <= 0.0 {
                continue;
            }
            let win_prob = qa / total;
            edges.push(Edge {
                item1: item_ids[a],
                item2: item_ids[b],
                category_probs: [win_prob, 1.0 - win_prob],
                slot1: a as u8,
                slot2: b as u8,
                judge_id,
                weight: 0.0, // set below
            });
        }
    }
    // Spread the judgement's degrees of freedom evenly over the edges that
    // survived. If every pair was dropped -- only possible when the winner
    // distribution is all zeros -- `m` is 0 and `w` is infinite, but `edges` is
    // then empty, so no infinite weight can reach the fit.
    let m = edges.len() as f64;
    let df = if logprobs_mode { (size - 1) as f64 } else { 1.0 };
    let w = df / m;
    for e in &mut edges {
        e.weight = w;
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win_prob(edge: &Edge) -> f64 {
        edge.category_probs[0]
    }

    #[test]
    fn soft_distribution_produces_three_luce_edges() {
        let edges = winner_dist_to_edges(&[10, 20, 30], &[0.9, 0.08, 0.02], 7, true);
        assert_eq!(edges.len(), 3);

        // A vs B: 0.9 / (0.9 + 0.08)
        assert_eq!((edges[0].item1, edges[0].item2), (10, 20));
        assert!((win_prob(&edges[0]) - 0.9 / 0.98).abs() < 1e-12);
        // A vs C: 0.9 / (0.9 + 0.02)
        assert_eq!((edges[1].item1, edges[1].item2), (10, 30));
        assert!((win_prob(&edges[1]) - 0.9 / 0.92).abs() < 1e-12);
        // B vs C: 0.08 / (0.08 + 0.02) = 0.8
        assert_eq!((edges[2].item1, edges[2].item2), (20, 30));
        assert!((win_prob(&edges[2]) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn category_probs_sum_to_one_and_are_consistent() {
        let edges = winner_dist_to_edges(&[1, 2, 3], &[0.5, 0.3, 0.2], 0, true);
        for e in &edges {
            let s: f64 = e.category_probs.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hard_one_hot_drops_the_losers_edge() {
        let edges = winner_dist_to_edges(&[1, 2, 3], &[1.0, 0.0, 0.0], 42, false);
        assert_eq!(edges.len(), 2);
        assert_eq!((edges[0].item1, edges[0].item2), (1, 2));
        assert!((win_prob(&edges[0]) - 1.0).abs() < 1e-12);
        assert_eq!((edges[1].item1, edges[1].item2), (1, 3));
        assert!((win_prob(&edges[1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn soft_edges_weighted_two_thirds() {
        let edges = winner_dist_to_edges(&[1, 2, 3], &[0.5, 0.3, 0.2], 0, true);
        assert_eq!(edges.len(), 3);
        for e in &edges {
            assert!((e.weight - 2.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hard_edges_weighted_one_half() {
        let edges = winner_dist_to_edges(&[1, 2, 3], &[1.0, 0.0, 0.0], 0, false);
        assert_eq!(edges.len(), 2);
        for e in &edges {
            assert!((e.weight - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn logprobs_one_hot_weighted_one() {
        // Logprobs mode with [1,0,0]: edge dropped but still 2 df.
        let edges = winner_dist_to_edges(&[1, 2, 3], &[1.0, 0.0, 0.0], 0, true);
        assert_eq!(edges.len(), 2);
        for e in &edges {
            assert!((e.weight - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn ratio_is_scale_invariant() {
        let a = winner_dist_to_edges(&[1, 2, 3], &[0.9, 0.08, 0.02], 0, true);
        let b = winner_dist_to_edges(&[1, 2, 3], &[9.0, 0.8, 0.2], 0, true);
        for (ea, eb) in a.iter().zip(b.iter()) {
            assert!((win_prob(ea) - win_prob(eb)).abs() < 1e-12);
        }
    }

    /// The df budget is `k − 1` in logprobs mode and 1 in text mode, spread over
    /// however many edges survive — at every supported lineup size.
    #[test]
    fn total_weight_matches_degrees_of_freedom_at_every_size() {
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let ids: Vec<i64> = (0..size as i64).collect();
            // A strictly decreasing, all-positive distribution: every pair survives.
            let probs: Vec<f64> = (0..size).map(|i| 1.0 / (i + 2) as f64).collect();

            let soft = winner_dist_to_edges(&ids, &probs, 0, true);
            assert_eq!(soft.len(), size * (size - 1) / 2);
            let soft_total: f64 = soft.iter().map(|e| e.weight).sum();
            assert!(
                (soft_total - (size - 1) as f64).abs() < 1e-12,
                "size {size}: logprobs total weight {soft_total} != {}",
                size - 1
            );

            let hard = winner_dist_to_edges(&ids, &probs, 0, false);
            let hard_total: f64 = hard.iter().map(|e| e.weight).sum();
            assert!(
                (hard_total - 1.0).abs() < 1e-12,
                "size {size}: text total weight {hard_total} != 1"
            );
        }
    }

    /// A nine-item one-hot verdict keeps only the winner's eight edges, and the
    /// df budget is unchanged by the drops.
    #[test]
    fn nine_item_one_hot_keeps_winner_edges_only() {
        let ids: Vec<i64> = (0..9).collect();
        let mut probs = [0.0_f64; 9];
        probs[3] = 1.0;
        let edges = winner_dist_to_edges(&ids, &probs, 0, true);
        assert_eq!(edges.len(), 8);
        for e in &edges {
            assert!(e.item1 == 3 || e.item2 == 3);
            assert!((e.weight - 8.0 / 8.0).abs() < 1e-12);
        }
    }

    /// Slots record presentation order, so per-slot bias correction sees all nine.
    #[test]
    fn slots_span_the_whole_lineup() {
        let ids: Vec<i64> = (0..9).collect();
        let probs: Vec<f64> = (0..9).map(|i| 1.0 / (i + 2) as f64).collect();
        let edges = winner_dist_to_edges(&ids, &probs, 0, true);
        let max_slot = edges.iter().map(|e| e.slot1.max(e.slot2)).max().unwrap();
        assert_eq!(max_slot, 8);
    }

    /// Size 2 reduces to a single ordinary pairwise edge of weight 1.
    #[test]
    fn size_two_is_one_unit_weight_edge() {
        let edges = winner_dist_to_edges(&[7, 9], &[0.75, 0.25], 3, true);
        assert_eq!(edges.len(), 1);
        assert_eq!((edges[0].item1, edges[0].item2), (7, 9));
        assert!((win_prob(&edges[0]) - 0.75).abs() < 1e-12);
        assert!((edges[0].weight - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn negative_winner_prob_panics() {
        winner_dist_to_edges(&[1, 2, 3], &[0.9, -0.1, 0.2], 0, true);
    }

    #[test]
    #[should_panic]
    fn duplicate_ids_panic() {
        winner_dist_to_edges(&[1, 1, 3], &[0.5, 0.3, 0.2], 0, true);
    }

    #[test]
    #[should_panic]
    fn mismatched_lengths_panic() {
        winner_dist_to_edges(&[1, 2, 3], &[0.5, 0.5], 0, true);
    }

    #[test]
    #[should_panic]
    fn oversized_lineup_panics() {
        let ids: Vec<i64> = (0..10).collect();
        let probs = vec![0.1_f64; 10];
        winner_dist_to_edges(&ids, &probs, 0, true);
    }
}
