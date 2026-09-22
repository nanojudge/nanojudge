//! Lineup judgement → edges.
//!
//! A judgement over a `k`-item lineup (`2 ≤ k ≤ 9`) is summarized as a
//! [`LineupVerdict`]: the judge's ranking, plus the probability it gave each
//! place's pick among the options still unplaced at that point. Under a Luce
//! model each place is a choice from the remaining options, with the chosen
//! option winning in proportion to its latent strength, so place `r`'s
//! probability is `s_r / (s_r + s_{r+1} + ... + s_{k-1})` (strengths indexed by
//! rank).
//!
//! The ranking engine consumes edges. This module converts one verdict into up
//! to `k · (k − 1) / 2` `Edge` values — one per unordered pair — via the Luce
//! ratio `P(i beats j) = s_i / (s_i + s_j)`. For the options ranked at places
//! `hi < lo`, the strengths relative to everything still unplaced at place `hi`
//! are
//!
//! ```text
//! s_hi = p_hi
//! s_lo = (1 − p_hi) · (1 − p_{hi+1}) · … · (1 − p_{lo−1}) · p_lo
//! ```
//!
//! (with `p_lo = 1` for the last place, which has no choice left). Each pair is
//! measured from the higher-ranked option's own place, so a certain pick at one
//! place never erases the comparisons below it: a lineup ranked A > B > C with
//! a certain 1st place still yields the B–C edge from the 2nd place's
//! probability. The engine then treats the edges as ordinary, mutually
//! consistent edges; the core likelihood is untouched.
//!
//! # Edge weighting
//!
//! A verdict carries `k − 1` degrees of freedom: each place but the last is one
//! choice. This holds in both modes — logprobs mode reads a probability at each
//! of those places, and text mode reads a hard pick at each (probability 1).
//! Each surviving edge gets `df / m` where `m` is the number of surviving edges:
//!
//! | k | Edges | df | Weight each | Total |
//! |---|-------|----|-------------|-------|
//! | 2 | 1     | 1  | 1           | 1     |
//! | 3 | 3     | 2  | 2/3         | 2     |
//! | 3 | 2     | 2  | 1           | 2     |
//! | 9 | 36    | 8  | 2/9         | 8     |

use crate::constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE};
use crate::types::Edge;

/// A parsed lineup judgement.
///
/// Indices refer to the lineup in presentation order (slot A is 0).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineupVerdict {
    /// The judge's ranking, best first: a permutation of `0..k`.
    pub ranking: Vec<usize>,
    /// `place_probs[r]` is the probability the judge gave `ranking[r]` at place
    /// `r`, among the options not yet placed above it. Length `k − 1`: the last
    /// place has only one option left, so it carries no probability. Text-mode
    /// verdicts are all 1.0.
    pub place_probs: Vec<f64>,
}

/// Convert a lineup verdict into edges.
///
/// `item_ids` are the caller IDs in presentation order; `verdict` indexes into
/// them. Produces up to one `Edge` per unordered pair, in index order (`(0,1)`,
/// `(0,2)`, ..., `(1,2)`, ...), each carrying `P(item1 beats item2)` from the
/// Luce ratio, all attributed to `judge_id`.
///
/// An edge is **dropped** when both its items have zero strength relative to
/// the higher-ranked one's place: the Luce ratio is then `0/0`, undefined,
/// carrying no information. That needs a pick the judge gave probability 0 —
/// a certain pick (probability 1) never drops anything.
///
/// # Panics
///
/// Panics if the lineup size is outside `2..=9`, if `verdict.ranking` is not a
/// permutation of the lineup's indices, if `verdict.place_probs` is not
/// `k − 1` long, if any place probability is not finite or lies outside
/// `[0, 1]`, or if the `item_ids` are not distinct.
pub fn lineup_verdict_to_edges(
    item_ids: &[i64],
    verdict: &LineupVerdict,
    judge_id: u64,
) -> Vec<Edge> {
    let size = item_ids.len();
    assert!(
        (MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE).contains(&size),
        "lineup size must be between {MIN_LINEUP_SIZE} and {MAX_LINEUP_SIZE}, got {size}"
    );
    assert_eq!(
        verdict.ranking.len(),
        size,
        "ranking ({}) must place every lineup item ({size})",
        verdict.ranking.len()
    );
    let mut place_of = vec![usize::MAX; size];
    for (place, &idx) in verdict.ranking.iter().enumerate() {
        assert!(
            idx < size && place_of[idx] == usize::MAX,
            "ranking must be a permutation of 0..{size}, got {:?}",
            verdict.ranking
        );
        place_of[idx] = place;
    }
    assert_eq!(
        verdict.place_probs.len(),
        size - 1,
        "place_probs ({}) must have one entry per place but the last ({})",
        verdict.place_probs.len(),
        size - 1
    );
    for &p in &verdict.place_probs {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "place_probs entries must be finite and in [0, 1], got {p}"
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

    // The last place has no choice left: it takes the whole remainder.
    let pick = |place: usize| -> f64 {
        if place == size - 1 { 1.0 } else { verdict.place_probs[place] }
    };

    let mut edges = Vec::with_capacity(size * (size - 1) / 2);
    for a in 0..size {
        for b in (a + 1)..size {
            let (hi, lo) = (place_of[a].min(place_of[b]), place_of[a].max(place_of[b]));
            // Strengths relative to everything still unplaced at place `hi`.
            let s_hi = pick(hi);
            let s_lo = (hi..lo).map(|t| 1.0 - pick(t)).product::<f64>() * pick(lo);
            let total = s_hi + s_lo;
            if total <= 0.0 {
                continue;
            }
            let hi_wins = s_hi / total;
            let win_prob = if place_of[a] == hi { hi_wins } else { 1.0 - hi_wins };
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
    // survived. If every pair was dropped `m` is 0 and `w` is infinite, but
    // `edges` is then empty, so no infinite weight can reach the fit.
    let m = edges.len() as f64;
    let df = (size - 1) as f64;
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

    fn verdict(ranking: &[usize], place_probs: &[f64]) -> LineupVerdict {
        LineupVerdict { ranking: ranking.to_vec(), place_probs: place_probs.to_vec() }
    }

    #[test]
    fn soft_verdict_produces_three_luce_edges() {
        // 1st A at 0.9, then B over C at 0.8: strengths 0.9 / 0.08 / 0.02.
        let edges = lineup_verdict_to_edges(&[10, 20, 30], &verdict(&[0, 1, 2], &[0.9, 0.8]), 7);
        assert_eq!(edges.len(), 3);

        // A vs B: 0.9 / (0.9 + 0.08)
        assert_eq!((edges[0].item1, edges[0].item2), (10, 20));
        assert!((win_prob(&edges[0]) - 0.9 / 0.98).abs() < 1e-12);
        // A vs C: 0.9 / (0.9 + 0.02)
        assert_eq!((edges[1].item1, edges[1].item2), (10, 30));
        assert!((win_prob(&edges[1]) - 0.9 / 0.92).abs() < 1e-12);
        // B vs C: the 2nd place's own probability
        assert_eq!((edges[2].item1, edges[2].item2), (20, 30));
        assert!((win_prob(&edges[2]) - 0.8).abs() < 1e-12);
        for e in &edges {
            assert_eq!(e.judge_id, 7);
        }
    }

    /// A certain 1st place must not erase the comparison below it: B vs C still
    /// comes from the 2nd place's probability.
    #[test]
    fn certain_first_place_keeps_the_lower_pair() {
        let edges = lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[1.0, 0.8]), 0);
        assert_eq!(edges.len(), 3);
        assert!((win_prob(&edges[0]) - 1.0).abs() < 1e-12);
        assert!((win_prob(&edges[1]) - 1.0).abs() < 1e-12);
        assert_eq!((edges[2].item1, edges[2].item2), (2, 3));
        assert!((win_prob(&edges[2]) - 0.8).abs() < 1e-12);
    }

    /// A hard (text-mode) ranking yields a hard edge for every pair, oriented
    /// by the ranking rather than by presentation order.
    #[test]
    fn hard_ranking_orients_every_pair() {
        // Presented A, B, C; ranked B > A > C.
        let edges = lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[1, 0, 2], &[1.0, 1.0]), 42);
        assert_eq!(edges.len(), 3);
        assert_eq!((edges[0].item1, edges[0].item2), (1, 2));
        assert_eq!(win_prob(&edges[0]), 0.0); // A loses to B
        assert_eq!((edges[1].item1, edges[1].item2), (1, 3));
        assert_eq!(win_prob(&edges[1]), 1.0); // A beats C
        assert_eq!((edges[2].item1, edges[2].item2), (2, 3));
        assert_eq!(win_prob(&edges[2]), 1.0); // B beats C
    }

    #[test]
    fn category_probs_sum_to_one() {
        let edges = lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[2, 0, 1], &[0.5, 0.6]), 0);
        for e in &edges {
            let s: f64 = e.category_probs.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn soft_edges_weighted_two_thirds() {
        let edges = lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[0.5, 0.6]), 0);
        assert_eq!(edges.len(), 3);
        for e in &edges {
            assert!((e.weight - 2.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hard_edges_weighted_two_thirds() {
        let edges = lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[1.0, 1.0]), 0);
        assert_eq!(edges.len(), 3);
        for e in &edges {
            assert!((e.weight - 2.0 / 3.0).abs() < 1e-12);
        }
    }

    /// A pick given probability 0 leaves the pair above it with no strength on
    /// either side: that edge is undefined and dropped, and the df budget is
    /// spread over the survivors.
    #[test]
    fn zero_probability_picks_drop_the_undefined_pair() {
        // 1st A at 0, 2nd B at 0: A and B both have zero strength beside C.
        let edges = lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[0.0, 0.0]), 0);
        assert_eq!(edges.len(), 2);
        assert_eq!((edges[0].item1, edges[0].item2), (1, 3));
        assert_eq!(win_prob(&edges[0]), 0.0);
        assert_eq!((edges[1].item1, edges[1].item2), (2, 3));
        assert_eq!(win_prob(&edges[1]), 0.0);
        for e in &edges {
            assert!((e.weight - 1.0).abs() < 1e-12);
        }
    }

    /// Whenever no place is certain, the edges equal the Luce ratios of the
    /// stick-breaking winner distribution (q for place r is its probability
    /// times everything left over by the places above it).
    #[test]
    fn matches_the_winner_distribution_when_nothing_is_certain() {
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let ids: Vec<i64> = (0..size as i64).collect();
            // A scrambled ranking (odd slots first, then even) and varied
            // place probabilities, none certain.
            let ranking: Vec<usize> = (1..size).step_by(2).chain((0..size).step_by(2)).collect();
            let place_probs: Vec<f64> = (0..size - 1).map(|r| 0.2 + 0.07 * r as f64).collect();

            let mut q = vec![0.0_f64; size];
            let mut residual = 1.0_f64;
            for (place, &idx) in ranking.iter().enumerate() {
                let p = if place == size - 1 { 1.0 } else { place_probs[place] };
                q[idx] = residual * p;
                residual -= q[idx];
            }

            let edges = lineup_verdict_to_edges(&ids, &verdict(&ranking, &place_probs), 0);
            assert_eq!(edges.len(), size * (size - 1) / 2);
            for e in &edges {
                let (a, b) = (e.item1 as usize, e.item2 as usize);
                let expected = q[a] / (q[a] + q[b]);
                assert!(
                    (win_prob(e) - expected).abs() < 1e-12,
                    "size {size}: edge ({a},{b}) = {}, expected {expected}",
                    win_prob(e)
                );
            }
        }
    }

    /// The df budget is `k − 1` at every supported lineup size, soft or hard.
    #[test]
    fn total_weight_matches_degrees_of_freedom_at_every_size() {
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let ids: Vec<i64> = (0..size as i64).collect();
            let ranking: Vec<usize> = (0..size).collect();
            for place_probs in [vec![0.5; size - 1], vec![1.0; size - 1]] {
                let edges = lineup_verdict_to_edges(&ids, &verdict(&ranking, &place_probs), 0);
                assert_eq!(edges.len(), size * (size - 1) / 2);
                let total: f64 = edges.iter().map(|e| e.weight).sum();
                assert!(
                    (total - (size - 1) as f64).abs() < 1e-12,
                    "size {size}: total weight {total} != {}",
                    size - 1
                );
            }
        }
    }

    /// A nine-item hard ranking keeps every pair, each oriented by rank.
    #[test]
    fn nine_item_hard_ranking_keeps_every_pair() {
        let ids: Vec<i64> = (0..9).collect();
        let ranking = vec![3, 8, 0, 5, 1, 7, 2, 6, 4];
        let edges = lineup_verdict_to_edges(&ids, &verdict(&ranking, &[1.0; 8]), 0);
        assert_eq!(edges.len(), 36);
        let place = |id: i64| ranking.iter().position(|&i| i as i64 == id).unwrap();
        for e in &edges {
            let expected = if place(e.item1) < place(e.item2) { 1.0 } else { 0.0 };
            assert_eq!(win_prob(e), expected);
            assert!((e.weight - 8.0 / 36.0).abs() < 1e-12);
        }
    }

    /// Slots record presentation order, so per-slot bias correction sees all nine.
    #[test]
    fn slots_span_the_whole_lineup() {
        let ids: Vec<i64> = (0..9).collect();
        let ranking: Vec<usize> = (0..9).collect();
        let edges = lineup_verdict_to_edges(&ids, &verdict(&ranking, &[0.5; 8]), 0);
        let max_slot = edges.iter().map(|e| e.slot1.max(e.slot2)).max().unwrap();
        assert_eq!(max_slot, 8);
    }

    /// Size 2 reduces to a single ordinary pairwise edge of weight 1.
    #[test]
    fn size_two_is_one_unit_weight_edge() {
        let edges = lineup_verdict_to_edges(&[7, 9], &verdict(&[0, 1], &[0.75]), 3);
        assert_eq!(edges.len(), 1);
        assert_eq!((edges[0].item1, edges[0].item2), (7, 9));
        assert!((win_prob(&edges[0]) - 0.75).abs() < 1e-12);
        assert!((edges[0].weight - 1.0).abs() < 1e-12);

        // Ranked the other way round, item1 wins with the complement.
        let edges = lineup_verdict_to_edges(&[7, 9], &verdict(&[1, 0], &[0.75]), 3);
        assert!((win_prob(&edges[0]) - 0.25).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn place_prob_above_one_panics() {
        lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[1.1, 0.5]), 0);
    }

    #[test]
    #[should_panic]
    fn negative_place_prob_panics() {
        lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[0.5, -0.1]), 0);
    }

    #[test]
    #[should_panic]
    fn non_permutation_ranking_panics() {
        lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 0, 2], &[0.5, 0.5]), 0);
    }

    #[test]
    #[should_panic]
    fn short_ranking_panics() {
        lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1], &[0.5, 0.5]), 0);
    }

    #[test]
    #[should_panic]
    fn wrong_place_probs_length_panics() {
        lineup_verdict_to_edges(&[1, 2, 3], &verdict(&[0, 1, 2], &[0.5, 0.5, 1.0]), 0);
    }

    #[test]
    #[should_panic]
    fn duplicate_ids_panic() {
        lineup_verdict_to_edges(&[1, 1, 3], &verdict(&[0, 1, 2], &[0.5, 0.5]), 0);
    }

    #[test]
    #[should_panic]
    fn oversized_lineup_panics() {
        let ids: Vec<i64> = (0..10).collect();
        let ranking: Vec<usize> = (0..10).collect();
        lineup_verdict_to_edges(&ids, &verdict(&ranking, &[0.5; 9]), 0);
    }
}
