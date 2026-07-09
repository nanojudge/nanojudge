//! Three-way comparison → pairwise edges.
//!
//! A 3-way judgment (show the judge three items, ask for a ranking) is summarized
//! as a **winner-distribution**: the probability each of the three items is the
//! best of the three (`q_A`, `q_B`, `q_C`). Under a Luce model this distribution
//! is the top-1 marginal, with `q_i ∝ s_i` (the items' latent strengths).
//!
//! The ranking engine speaks pairwise. This module converts one winner-distribution
//! into up to three pairwise `ComparisonInput` edges — one per unordered pair — via
//! the Luce ratio `P(i beats j) = q_i / (q_i + q_j)`. The engine then treats them
//! as three ordinary (mutually consistent) pairwise comparisons; the core likelihood
//! is untouched.
//!
//! Because the ratio only depends on `q_i / q_j`, the winner-distribution need not
//! be exactly normalized — any positive scaling gives the same edges.
//!
//! # Known approximation
//!
//! The three edges come from a single observation but enter the likelihood as if
//! independent, so a 3-way judgment contributes slightly more apparent information
//! than it truly carries — credible intervals run a touch tight. This is acceptable
//! for ranking accuracy; a native Plackett-Luce likelihood would be the exact fix.

use crate::types::ComparisonInput;

/// The three unordered pairs of a 3-item set, as index pairs into a `[_; 3]`.
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Convert a 3-way winner-distribution into pairwise comparison edges.
///
/// `item_ids` are the three caller IDs; `winner_probs[k]` is the probability that
/// `item_ids[k]` is the best of the three. Produces up to three `ComparisonInput`s
/// (pairs `(0,1)`, `(0,2)`, `(1,2)`), each carrying `P(item1 beats item2)` from the
/// Luce ratio, all attributed to `judge_id`.
///
/// An edge is **dropped** when both its items have zero winner-probability: the Luce
/// ratio is then `0/0`, undefined, carrying no information. This is exactly the
/// non-logprobs (hard) case — a one-hot winner-distribution like `(1, 0, 0)` yields
/// the winner's two edges (hard wins) and drops the losers' edge.
///
/// # Panics
///
/// Panics if any `winner_probs` entry is not finite or is negative, or if the three
/// `item_ids` are not distinct.
pub fn winner_dist_to_edges(
    item_ids: [i64; 3],
    winner_probs: [f64; 3],
    judge_id: u64,
) -> Vec<ComparisonInput> {
    for &q in &winner_probs {
        assert!(
            q.is_finite() && q >= 0.0,
            "winner_probs entries must be finite and non-negative, got {q}"
        );
    }
    assert!(
        item_ids[0] != item_ids[1] && item_ids[0] != item_ids[2] && item_ids[1] != item_ids[2],
        "three-way item_ids must be distinct, got {item_ids:?}"
    );

    let mut edges = Vec::with_capacity(3);
    for &(a, b) in &PAIRS {
        let (qa, qb) = (winner_probs[a], winner_probs[b]);
        let total = qa + qb;
        if total <= 0.0 {
            // Both items have zero winner-probability: 0/0 edge, no information.
            continue;
        }
        let win_prob = qa / total;
        edges.push(ComparisonInput {
            item1: item_ids[a],
            item2: item_ids[b],
            // Only p[0]+p[1] is read downstream, so a two-bucket split suffices.
            category_probs: [win_prob, 0.0, 0.0, 1.0 - win_prob],
            // The PAIRS indices are the presentation slots (0=A, 1=B, 2=C), so the
            // scoring engine can estimate and correct each slot's positional bias.
            slot1: a as u8,
            slot2: b as u8,
            judge_id,
        });
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win_prob(edge: &ComparisonInput) -> f64 {
        edge.category_probs[0] + edge.category_probs[1]
    }

    #[test]
    fn soft_distribution_produces_three_luce_edges() {
        let edges = winner_dist_to_edges([10, 20, 30], [0.9, 0.08, 0.02], 7);
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
        let edges = winner_dist_to_edges([1, 2, 3], [0.5, 0.3, 0.2], 0);
        for e in &edges {
            let s: f64 = e.category_probs.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hard_one_hot_drops_the_losers_edge() {
        // Non-logprobs mode: winner takes all.
        let edges = winner_dist_to_edges([1, 2, 3], [1.0, 0.0, 0.0], 42);
        assert_eq!(edges.len(), 2);
        // Both surviving edges are hard wins for the winner (item 1).
        assert_eq!((edges[0].item1, edges[0].item2), (1, 2));
        assert!((win_prob(&edges[0]) - 1.0).abs() < 1e-12);
        assert_eq!((edges[1].item1, edges[1].item2), (1, 3));
        assert!((win_prob(&edges[1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ratio_is_scale_invariant() {
        let a = winner_dist_to_edges([1, 2, 3], [0.9, 0.08, 0.02], 0);
        let b = winner_dist_to_edges([1, 2, 3], [9.0, 0.8, 0.2], 0);
        for (ea, eb) in a.iter().zip(b.iter()) {
            assert!((win_prob(ea) - win_prob(eb)).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic]
    fn negative_winner_prob_panics() {
        winner_dist_to_edges([1, 2, 3], [0.9, -0.1, 0.2], 0);
    }

    #[test]
    #[should_panic]
    fn duplicate_ids_panic() {
        winner_dist_to_edges([1, 1, 3], [0.5, 0.3, 0.2], 0);
    }
}
