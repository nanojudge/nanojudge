//! Three-item lineup judgement → edges.
//!
//! A judgement over a three-item lineup is summarized
//! as a **winner-distribution**: the probability each of the three items is the
//! best of the three (`q_A`, `q_B`, `q_C`). Under a Luce model this distribution
//! is the top-1 marginal, with `q_i ∝ s_i` (the items' latent strengths).
//!
//! The ranking engine consumes edges. This module converts one winner distribution
//! into up to three `Edge` values — one per unordered pair — via
//! the Luce ratio `P(i beats j) = q_i / (q_i + q_j)`. The engine then treats them
//! as three ordinary, mutually consistent edges; the core likelihood
//! is untouched.
//!
//! Because the ratio only depends on `q_i / q_j`, the winner-distribution need not
//! be exactly normalized — any positive scaling gives the same edges.
//!
//! # Edge weighting
//!
//! The degrees of freedom depend on the verdict's provenance, not on how many edges
//! survive. Logprobs mode always carries 2 df (the full winner-distribution has 2
//! free parameters). Text mode carries 1 df (a single categorical "who won").
//! Each surviving edge gets `df / k` where `k` is the number of surviving edges:
//!
//! | Mode     | Edges | df | Weight each | Total |
//! |----------|-------|----|-------------|-------|
//! | Logprobs | 3     | 2  | 2/3         | 2     |
//! | Logprobs | 2     | 2  | 1           | 2     |
//! | Text     | 2     | 1  | 1/2         | 1     |

use crate::types::Edge;

/// The three unordered pairs of a 3-item set, as index pairs into a `[_; 3]`.
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Convert a three-item lineup's winner distribution into edges.
///
/// `item_ids` are the three caller IDs; `winner_probs[k]` is the probability that
/// `item_ids[k]` is the best of the three. Produces up to three `Edge`s
/// (pairs `(0,1)`, `(0,2)`, `(1,2)`), each carrying `P(item1 beats item2)` from the
/// Luce ratio, all attributed to `judge_id`.
///
/// An edge is **dropped** when both its items have zero winner-probability: the Luce
/// ratio is then `0/0`, undefined, carrying no information.
///
/// `logprobs_mode` controls the degrees of freedom: `true` means the verdict came
/// from a full probability distribution (2 df even if an edge is dropped because a
/// probability rounded to zero), `false` means a text-mode winner-only verdict (1 df).
///
/// # Panics
///
/// Panics if any `winner_probs` entry is not finite or is negative, or if the three
/// `item_ids` are not distinct.
pub fn winner_dist_to_edges(
    item_ids: [i64; 3],
    winner_probs: [f64; 3],
    judge_id: u64,
    logprobs_mode: bool,
) -> Vec<Edge> {
    for &q in &winner_probs {
        assert!(
            q.is_finite() && q >= 0.0,
            "winner_probs entries must be finite and non-negative, got {q}"
        );
    }
    assert!(
        item_ids[0] != item_ids[1] && item_ids[0] != item_ids[2] && item_ids[1] != item_ids[2],
        "lineup item_ids must be distinct, got {item_ids:?}"
    );

    let mut edges = Vec::with_capacity(3);
    for &(a, b) in &PAIRS {
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
    let k = edges.len() as f64;
    let df = if logprobs_mode { 2.0 } else { 1.0 };
    let w = df / k;
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
        let edges = winner_dist_to_edges([10, 20, 30], [0.9, 0.08, 0.02], 7, true);
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
        let edges = winner_dist_to_edges([1, 2, 3], [0.5, 0.3, 0.2], 0, true);
        for e in &edges {
            let s: f64 = e.category_probs.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hard_one_hot_drops_the_losers_edge() {
        let edges = winner_dist_to_edges([1, 2, 3], [1.0, 0.0, 0.0], 42, false);
        assert_eq!(edges.len(), 2);
        assert_eq!((edges[0].item1, edges[0].item2), (1, 2));
        assert!((win_prob(&edges[0]) - 1.0).abs() < 1e-12);
        assert_eq!((edges[1].item1, edges[1].item2), (1, 3));
        assert!((win_prob(&edges[1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn soft_edges_weighted_two_thirds() {
        let edges = winner_dist_to_edges([1, 2, 3], [0.5, 0.3, 0.2], 0, true);
        assert_eq!(edges.len(), 3);
        for e in &edges {
            assert!((e.weight - 2.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hard_edges_weighted_one_half() {
        let edges = winner_dist_to_edges([1, 2, 3], [1.0, 0.0, 0.0], 0, false);
        assert_eq!(edges.len(), 2);
        for e in &edges {
            assert!((e.weight - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn logprobs_one_hot_weighted_one() {
        // Logprobs mode with [1,0,0]: edge dropped but still 2 df.
        let edges = winner_dist_to_edges([1, 2, 3], [1.0, 0.0, 0.0], 0, true);
        assert_eq!(edges.len(), 2);
        for e in &edges {
            assert!((e.weight - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn ratio_is_scale_invariant() {
        let a = winner_dist_to_edges([1, 2, 3], [0.9, 0.08, 0.02], 0, true);
        let b = winner_dist_to_edges([1, 2, 3], [9.0, 0.8, 0.2], 0, true);
        for (ea, eb) in a.iter().zip(b.iter()) {
            assert!((win_prob(ea) - win_prob(eb)).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic]
    fn negative_winner_prob_panics() {
        winner_dist_to_edges([1, 2, 3], [0.9, -0.1, 0.2], 0, true);
    }

    #[test]
    #[should_panic]
    fn duplicate_ids_panic() {
        winner_dist_to_edges([1, 1, 3], [0.5, 0.3, 0.2], 0, true);
    }
}
