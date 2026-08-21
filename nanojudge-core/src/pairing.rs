/// Judgement distributions for ranking tournaments.
///
/// Public functions accept `item_ids: &[i64]` and return `Pair` (i64, i64).
/// Internal functions use `usize` indices for efficient array indexing.
use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE, OPPONENT_WINDOW_SIZE};
use crate::seed::make_rng;
use crate::types::{IndexedPair, IndexedLineup, Pair};

/// Judgement distribution enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum JudgementDistribution {
    Uniform,
    TopHeavy,
}

/// Calculate information gain for a matchup between two items.
pub fn calculate_info_gain(rating_a: f64, rating_b: f64, sharpness: f64) -> f64 {
    let p = 1.0 / (1.0 + (rating_b - rating_a).exp());
    let info_gain = p * (1.0 - p);
    info_gain.powf(sharpness)
}

/// Information gain for a matchup, integrating the win probability over
/// Gaussian rating uncertainty: with independent ratings ~ N(rating, std²),
/// the expected win probability is `sigmoid(gap / kappa)` where
/// `kappa = sqrt(1 + pi * (std_a² + std_b²) / 8)` (the logistic-probit
/// bridge). Shrinking the gap this way can only pull p toward 0.5, so
/// uncertain matchups — pairs that *might* be close — gain weight over their
/// point estimates. With both stds 0, `kappa` is exactly 1 and this reduces
/// bit-for-bit to `calculate_info_gain`.
pub fn calculate_integrated_info_gain(
    rating_a: f64,
    std_a: f64,
    rating_b: f64,
    std_b: f64,
    sharpness: f64,
) -> f64 {
    let kappa = (1.0 + std::f64::consts::PI * (std_a * std_a + std_b * std_b) / 8.0).sqrt();
    let p = 1.0 / (1.0 + ((rating_b - rating_a) / kappa).exp());
    (p * (1.0 - p)).powf(sharpness)
}

/// Opponent-window gain used by every window loop below: integrated info gain
/// when per-item posterior stds are available (top-heavy runs push them in
/// from each interim fit), plain plug-in info gain otherwise (uniform runs,
/// where only the MLE point estimate exists).
fn window_info_gain(
    rating_a: f64,
    rating_b: f64,
    item_a: usize,
    item_b: usize,
    current_stds: Option<&[f64]>,
    sharpness: f64,
) -> f64 {
    match current_stds {
        Some(stds) => {
            calculate_integrated_info_gain(rating_a, stds[item_a], rating_b, stds[item_b], sharpness)
        }
        None => calculate_info_gain(rating_a, rating_b, sharpness),
    }
}

/// Probability that item A goes in position 1 against item B.
///
/// Uses Laplace-smoothed first-position ratios so that an item which has
/// historically gone first more often gets a lower probability of going
/// first next time. With no history (zero counts) this returns 0.5,
/// degrading gracefully to a fair coin flip. The +1 / +2 smoothing damps
/// extreme ratios from small samples.
fn position_probability(
    first_count_a: usize, edges_a: usize,
    first_count_b: usize, edges_b: usize,
) -> f64 {
    let ratio_a = (first_count_a as f64 + 1.0) / (edges_a as f64 + 2.0);
    let ratio_b = (first_count_b as f64 + 1.0) / (edges_b as f64 + 2.0);
    ratio_b / (ratio_a + ratio_b)
}

/// Determine the effective judgement distribution to use before the next refit.
///
/// Two stages:
///   Stage 1 (Uniform): uniform pairing until every item has >= min_uniform_edges.
///   Stage 2 (Top-heavy): top-heavy pairing thereafter.
///
/// # Panics
///
/// Panics if `edge_counts` has fewer than `num_items` entries.
pub fn get_effective_judgement_distribution(
    user_judgement_distribution: JudgementDistribution,
    num_items: usize,
    edge_counts: &[usize],
    min_uniform_edges: usize,
) -> JudgementDistribution {
    if user_judgement_distribution == JudgementDistribution::Uniform {
        return JudgementDistribution::Uniform;
    }

    // Stage 1: uniform until every item has >= min_uniform_edges.
    for &edges in edge_counts.iter().take(num_items) {
        if edges < min_uniform_edges {
            return JudgementDistribution::Uniform;
        }
    }

    // Stage 2: top-heavy.
    JudgementDistribution::TopHeavy
}

// ---------------------------------------------------------------------------
// Public pairing functions (work with i64 IDs)
// ---------------------------------------------------------------------------

/// Generate the requested number of cumulatively balanced uniform pairings.
///
/// `current_ratings[i]`, `item1_edge_counts[i]` and `edge_counts[i]` all
/// correspond to `item_ids[i]`. Returns pairs of item IDs.
///
/// The minimum cumulative `edge_counts` value determines the matchmaking stage:
/// while any item has no edges, items are paired at random; once every item has
/// at least one edge, nearest-rating neighbours are paired; once every item has
/// at least two edges, opponents are drawn from a rating window weighted by info
/// gain (`sharpness` is the info-gain exponent). `item1_edge_counts` balances
/// which item of each pair is listed first. Callers must pass their real
/// cumulative counts.
///
/// `current_stds[i]` (optional) is the posterior std of `current_ratings[i]`;
/// when supplied, opponent gain integrates over that uncertainty
/// (`calculate_integrated_info_gain`) instead of using the point estimates.
///
/// # Panics
///
/// Panics if `current_ratings`, `current_stds` (when supplied),
/// `item1_edge_counts` or `edge_counts` length does not equal `item_ids`
/// length.
pub fn generate_uniform_pairings(
    item_ids: &[i64],
    pairs_count: usize,
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    sharpness: f64,
    item1_edge_counts: &[usize],
    edge_counts: &[usize],
) -> Vec<Pair> {
    let n = item_ids.len();
    assert_eq!(current_ratings.len(), n, "current_ratings length mismatch");
    if let Some(stds) = current_stds {
        assert_eq!(stds.len(), n, "current_stds length mismatch");
    }
    assert_eq!(item1_edge_counts.len(), n, "item1_edge_counts length mismatch");
    assert_eq!(edge_counts.len(), n, "edge_counts length mismatch");
    let mut rng = make_rng(None, crate::seed::SUBSYSTEM_PAIRING);
    let index_pairs = generate_uniform_pairings_indexed(
        n,
        pairs_count,
        current_ratings,
        current_stds,
        sharpness,
        item1_edge_counts,
        edge_counts,
        &mut rng,
    );
    index_pairs.into_iter().map(|(a, b)| (item_ids[a], item_ids[b])).collect()
}

/// Generate the requested number of top-heavy pairings.
///
/// item1 of every pair is sampled from the per-item selection weights
/// `selection_weights[i]` (corresponding to `item_ids[i]`), concentrating
/// judgements on the contenders. The opponent (item2) is then chosen by
/// info-gain matchmaking from a rating window around item1 (using
/// `current_ratings` and `matchmaking_sharpness`), exactly as uniform pairing
/// selects opponents. `item1_edge_counts[i]` and `edge_counts[i]`
/// (cumulative counts for `item_ids[i]`) balance which item of each pair is
/// listed first. Returns pairs of item IDs.
///
/// `current_stds[i]` (optional) is the posterior std of `current_ratings[i]`;
/// when supplied, opponent gain integrates over that uncertainty
/// (`calculate_integrated_info_gain`) instead of using the point estimates.
///
/// # Panics
///
/// Panics if `selection_weights`, `current_ratings`, `current_stds` (when
/// supplied), `item1_edge_counts` or `edge_counts` length does not equal
/// `item_ids` length, or if the total selection weight is not positive.
#[allow(clippy::too_many_arguments)]
pub fn generate_top_heavy_pairings(
    item_ids: &[i64],
    pairs_count: usize,
    selection_weights: &[f64],
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    matchmaking_sharpness: f64,
    item1_edge_counts: &[usize],
    edge_counts: &[usize],
) -> Vec<Pair> {
    let n = item_ids.len();
    assert_eq!(item1_edge_counts.len(), n, "item1_edge_counts length mismatch");
    assert_eq!(edge_counts.len(), n, "edge_counts length mismatch");
    let mut rng = make_rng(None, crate::seed::SUBSYSTEM_PAIRING);
    let index_pairs = generate_top_heavy_pairings_indexed(
        n,
        pairs_count,
        selection_weights,
        current_ratings,
        current_stds,
        matchmaking_sharpness,
        item1_edge_counts,
        edge_counts,
        &mut rng,
    );
    index_pairs.into_iter().map(|(a, b)| (item_ids[a], item_ids[b])).collect()
}

// ---------------------------------------------------------------------------
// Internal indexed pairing functions (work with usize indices)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_uniform_pairings_indexed(
    num_items: usize,
    pairs_count: usize,
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    sharpness: f64,
    item1_edge_counts: &[usize],
    edge_counts: &[usize],
    rng: &mut StdRng,
) -> Vec<IndexedPair> {
    let mut pairings: Vec<IndexedPair> = Vec::with_capacity(pairs_count);

    if num_items < 2 {
        return pairings;
    }

    // Local mutable copies of caller-provided counters. Updated optimistically
    // as positions are assigned within this call so later pairs balance against
    // earlier ones. Discarded at function return — caller state is never mutated.
    let mut local_first_counts: Vec<usize> = item1_edge_counts.to_vec();
    let mut local_edge_counts: Vec<usize> = edge_counts.to_vec();

    let judgements_to_include_every_item_once = num_items.div_ceil(2);
    while pairings.len() < pairs_count {
        let iteration_size =
            (pairs_count - pairings.len()).min(judgements_to_include_every_item_once);
        let before = pairings.len();
        generate_uniform_iteration(
            num_items,
            current_ratings,
            current_stds,
            sharpness,
            iteration_size,
            &mut pairings,
            rng,
            &mut local_first_counts,
            &mut local_edge_counts,
        );
        assert!(pairings.len() > before, "uniform pair generation made no progress");
    }

    pairings
}

/// Pick the items with the fewest incident edges, randomising ties. If a request
/// ends before every item can be selected, the next call therefore prioritises
/// the items that were not selected.
fn least_edge_count_pool(edge_counts: &[usize], slots: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..edge_counts.len()).collect();
    pool.shuffle(rng);
    pool.sort_by_key(|&item| edge_counts[item]);
    pool.truncate(slots.min(pool.len()));
    pool
}

#[allow(clippy::too_many_arguments)]
fn generate_uniform_iteration(
    num_items: usize,
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    sharpness: f64,
    max_pairs: usize,
    pairings: &mut Vec<IndexedPair>,
    rng: &mut impl Rng,
    first_counts: &mut [usize],
    total_edge_counts: &mut [usize],
) {
    // The minimum cumulative incident-edge count determines which matchmaking
    // strategy to use. For pair judgements, one appearance produces one edge:
    //
    //   0 edges: no complete baseline — pair uniformly at random.
    //   1 edge:  coarse ratings — pair nearest-strength neighbours.
    //   2+ edges: richer ratings — use info-gain matchmaking.
    let min_edge_count = total_edge_counts.iter().copied().min().unwrap_or(0);
    let slots = max_pairs.saturating_mul(2).min(num_items);
    let pool = least_edge_count_pool(total_edge_counts, slots, rng);
    let mut unoriented: Vec<(usize, usize)> = if min_edge_count == 0 {
        random_pairs(&pool, max_pairs, rng)
    } else if min_edge_count == 1 {
        nearest_neighbour_pairs(&pool, current_ratings, max_pairs, rng)
    } else {
        info_gain_pairs(&pool, current_ratings, current_stds, sharpness, max_pairs, rng)
    };

    // A non-divisible item count leaves one item after the disjoint pairs. The
    // final pair reuses one selected item so every item is included; later
    // iterations favour items with fewer edges and rebalance the difference.
    if unoriented.len() < max_pairs {
        let mut seen = vec![false; num_items];
        for &(a, b) in &unoriented {
            seen[a] = true;
            seen[b] = true;
        }
        let leftover = pool
            .iter()
            .copied()
            .find(|&item| !seen[item])
            .expect("incomplete pair iteration must have an unselected item");
        let mut candidates: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|&item| item != leftover)
            .collect();
        candidates.shuffle(rng);
        let partner = if min_edge_count == 0 {
            candidates[0]
        } else if min_edge_count == 1 {
            candidates
                .into_iter()
                .min_by(|&a, &b| {
                    (current_ratings[leftover] - current_ratings[a])
                        .abs()
                        .partial_cmp(&(current_ratings[leftover] - current_ratings[b]).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("pair completion requires an opponent")
        } else {
            let weights: Vec<f64> = candidates
                .iter()
                .map(|&item| {
                    window_info_gain(
                        current_ratings[leftover],
                        current_ratings[item],
                        leftover,
                        item,
                        current_stds,
                        sharpness,
                    )
                })
                .collect();
            let total_weight: f64 = weights.iter().sum();
            let selected = if total_weight == 0.0 {
                rng.random_range(0..candidates.len())
            } else {
                weighted_random_select(&weights, total_weight, rng)
            };
            candidates[selected]
        };
        unoriented.push((leftover, partner));
    }

    // Assign each pair an orientation (which item goes in position 1), balancing
    // first-position counts, then record it.
    for (item1, item2) in unoriented {
        let p = position_probability(
            first_counts[item1], total_edge_counts[item1],
            first_counts[item2], total_edge_counts[item2],
        );
        if rng.random::<f64>() < p {
            pairings.push((item1, item2));
            first_counts[item1] += 1;
        } else {
            pairings.push((item2, item1));
            first_counts[item2] += 1;
        }
        total_edge_counts[item1] += 1;
        total_edge_counts[item2] += 1;
    }
}

/// While some items have no edges, pair the selected items at random.
fn random_pairs(item_pool: &[usize], max_pairs: usize, rng: &mut impl Rng) -> Vec<(usize, usize)> {
    let mut order = item_pool.to_vec();
    order.shuffle(rng);
    order
        .chunks_exact(2)
        .take(max_pairs)
        .map(|c| (c[0], c[1]))
        .collect()
}

/// Once every item has an edge, sort by rating and pair adjacent neighbours —
/// the closest-strength opponent is simply the next item in sorted order.
/// Shuffle before the stable sort so equal-rated items are ordered randomly
/// rather than by index.
fn nearest_neighbour_pairs(
    item_pool: &[usize],
    current_ratings: &[f64],
    max_pairs: usize,
    rng: &mut impl Rng,
) -> Vec<(usize, usize)> {
    let mut pool: Vec<(usize, f64)> = item_pool
        .iter()
        .map(|&i| (i, current_ratings[i]))
        .collect();
    pool.shuffle(rng);
    pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    pool.chunks_exact(2)
        .take(max_pairs)
        .map(|c| (c[0].0, c[1].0))
        .collect()
}

/// Once every item has at least two edges, use info-gain matchmaking. Sort by
/// rating, then repeatedly pick a random still-unpaired item1 and draw its
/// opponent from a rating window around it, weighted by information gain so
/// closely-matched (more informative) pairs are favoured. Both items are removed
/// once paired.
fn info_gain_pairs(
    item_pool: &[usize],
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    sharpness: f64,
    max_pairs: usize,
    rng: &mut impl Rng,
) -> Vec<(usize, usize)> {
    // Sort items by rating ascending so opponents can be picked from a narrow
    // rating window around each item1. sorted_pool is immutable for the rest of
    // the function; "removal" is done via tombstones so the remaining entries
    // keep their sorted positions (and the window math stays valid).
    let mut sorted_pool: Vec<(usize, f64)> = item_pool
        .iter()
        .map(|&i| (i, current_ratings[i]))
        .collect();
    sorted_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_pool.len();
    // live_positions holds the sorted-pool indices still available. Picking
    // item1 is a swap_remove on this list (O(1) random removal). live_idx_of is
    // the inverse map: live_idx_of[p] = index of p inside live_positions, or
    // None if tombstoned. It lets us also O(1)-remove item2 once we know its
    // sorted-pool position.
    let mut live_positions: Vec<usize> = (0..n).collect();
    let mut live_idx_of: Vec<Option<usize>> = (0..n).map(Some).collect();
    // Rating-order links let an exhausted fixed window reach the nearest live
    // entries outside it without scanning across an unbounded run of tombstones.
    let (mut previous_live, mut next_live) = make_live_neighbour_links(n);

    let mut weights: Vec<f64> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 2);
    let mut candidates: Vec<usize> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 2);

    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(max_pairs);

    for _ in 0..max_pairs {
        if live_positions.len() < 2 {
            break;
        }

        // Pick item1: random live entry, removed via swap_remove.
        let live_idx1 = rng.random_range(0..live_positions.len());
        let (pos1, left, right) = remove_live_position(
            &mut live_positions,
            &mut live_idx_of,
            &mut previous_live,
            &mut next_live,
            live_idx1,
        );
        let (item1, item1_rating) = sorted_pool[pos1];

        // Use the normal fixed rating window. If it has been exhausted by
        // earlier pairings, extend only far enough to reach one live opponent.
        collect_live_window_candidates(
            pos1,
            left,
            right,
            &previous_live,
            &next_live,
            n,
            1,
            &mut candidates,
        );
        weights.clear();
        for &p in &candidates {
            weights.push(window_info_gain(
                item1_rating,
                sorted_pool[p].1,
                item1,
                sorted_pool[p].0,
                current_stds,
                sharpness,
            ));
        }

        let total_weight: f64 = weights.iter().sum();
        let selected = if total_weight == 0.0 {
            rng.random_range(0..candidates.len())
        } else {
            weighted_random_select(&weights, total_weight, rng)
        };

        let pos2 = candidates[selected];
        let live_idx2 = live_idx_of[pos2].expect("candidate must be alive");
        remove_live_position(
            &mut live_positions,
            &mut live_idx_of,
            &mut previous_live,
            &mut next_live,
            live_idx2,
        );
        let (item2, _) = sorted_pool[pos2];

        pairs.push((item1, item2));
    }

    pairs
}

/// Build a doubly linked list over the rating-sorted positions.
fn make_live_neighbour_links(n: usize) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let previous = (0..n).map(|pos| pos.checked_sub(1)).collect();
    let next = (0..n)
        .map(|pos| if pos + 1 < n { Some(pos + 1) } else { None })
        .collect();
    (previous, next)
}

/// Remove one live entry in O(1), keeping both the random-selection pool and
/// the rating-order neighbour links in sync. Returns the removed sorted-pool
/// position and the live neighbours it had immediately before removal.
fn remove_live_position(
    live_positions: &mut Vec<usize>,
    live_idx_of: &mut [Option<usize>],
    previous_live: &mut [Option<usize>],
    next_live: &mut [Option<usize>],
    live_idx: usize,
) -> (usize, Option<usize>, Option<usize>) {
    let removed_pos = live_positions.swap_remove(live_idx);
    live_idx_of[removed_pos] = None;
    // If we didn't remove the last entry, the old last entry now sits at live_idx.
    if live_idx < live_positions.len() {
        let moved_pos = live_positions[live_idx];
        live_idx_of[moved_pos] = Some(live_idx);
    }

    let previous = previous_live[removed_pos];
    let next = next_live[removed_pos];
    if let Some(previous_pos) = previous {
        next_live[previous_pos] = next;
    }
    if let Some(next_pos) = next {
        previous_live[next_pos] = previous;
    }

    (removed_pos, previous, next)
}

/// Collect live opponents in the existing fixed positional window around an
/// anchor. If tombstones leave fewer than `minimum` candidates there, follow
/// the live-neighbour links outward only until the minimum is reached.
/// Candidates remain in ascending rating-position order, preserving the
/// existing weighted-selection behavior whenever the fixed window suffices.
#[allow(clippy::too_many_arguments)]
fn collect_live_window_candidates(
    anchor_pos: usize,
    mut left: Option<usize>,
    mut right: Option<usize>,
    previous_live: &[Option<usize>],
    next_live: &[Option<usize>],
    n: usize,
    minimum: usize,
    candidates: &mut Vec<usize>,
) {
    let half_w = OPPONENT_WINDOW_SIZE / 2;
    let window_start = anchor_pos.saturating_sub(half_w);
    let window_end = (anchor_pos + half_w + 1).min(n);

    candidates.clear();

    // Walking left discovers candidates in reverse rating order. Reverse that
    // portion before appending the ascending right-hand side.
    while let Some(pos) = left {
        if pos < window_start {
            break;
        }
        candidates.push(pos);
        left = previous_live[pos];
    }
    candidates.reverse();

    while let Some(pos) = right {
        if pos >= window_end {
            break;
        }
        candidates.push(pos);
        right = next_live[pos];
    }

    // The fixed window is normally sufficient. In the rare exhausted case,
    // choose the closest remaining rating-order neighbour on either side and
    // stop as soon as the judgement can be completed.
    while candidates.len() < minimum {
        let take_left = match (left, right) {
            (Some(left_pos), Some(right_pos)) => {
                anchor_pos.abs_diff(left_pos) <= anchor_pos.abs_diff(right_pos)
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if take_left {
            let pos = left.expect("left live neighbour must exist");
            candidates.insert(0, pos);
            left = previous_live[pos];
        } else {
            let pos = right.expect("right live neighbour must exist");
            candidates.push(pos);
            right = next_live[pos];
        }
    }

    assert!(
        candidates.len() >= minimum,
        "live pool did not contain the required number of opponents"
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_top_heavy_pairings_indexed(
    num_items: usize,
    pairs_count: usize,
    selection_weights: &[f64],
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    matchmaking_sharpness: f64,
    item1_edge_counts: &[usize],
    edge_counts: &[usize],
    rng: &mut StdRng,
) -> Vec<IndexedPair> {
    if num_items < 2 {
        return Vec::new();
    }
    assert_eq!(selection_weights.len(), num_items, "selection_weights length mismatch");
    assert_eq!(current_ratings.len(), num_items, "current_ratings length mismatch");
    if let Some(stds) = current_stds {
        assert_eq!(stds.len(), num_items, "current_stds length mismatch");
    }
    let pairs_target = pairs_count;

    let total_weight: f64 = selection_weights.iter().sum();
    assert!(
        total_weight > 0.0,
        "top-heavy pairing requires positive total item-selection weight"
    );

    // Sort items by rating ascending so each item1's opponent can be drawn from a
    // narrow rating window around it — the same info-gain matchmaking uniform
    // uses. The pool is built once per call; item1 is sampled with replacement
    // from the selection weights, so no entries are removed between pairs.
    let mut sorted_pool: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, current_ratings[i]))
        .collect();
    sorted_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    // Inverse map: item index -> its position in sorted_pool.
    let mut sorted_position = vec![0usize; num_items];
    for (pos, &(item, _)) in sorted_pool.iter().enumerate() {
        sorted_position[item] = pos;
    }

    let half_w = OPPONENT_WINDOW_SIZE / 2;
    let mut candidates: Vec<usize> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 1);
    let mut weights: Vec<f64> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 1);

    // Local mutable copies of caller-provided counters. Updated optimistically
    // as positions are assigned within this call so later pairs balance against
    // earlier ones. Discarded at function return — caller state is never mutated.
    let mut local_first_counts: Vec<usize> = item1_edge_counts.to_vec();
    let mut local_edge_counts: Vec<usize> = edge_counts.to_vec();

    let mut pairs: Vec<IndexedPair> = Vec::with_capacity(pairs_target);

    for _ in 0..pairs_target {
        // item1: sampled from the global selection weights, concentrating
        // judgements on the contenders.
        let item1 = weighted_random_select(selection_weights, total_weight, rng);
        let item1_rating = current_ratings[item1];
        let pos1 = sorted_position[item1];

        // item2 (the opponent): the info-gain pick from a rating window around
        // item1, identical to how uniform pairing chooses an opponent. Closely
        // matched items carry more information, so the window concentrates on
        // them; item1 itself is excluded so the pair is always distinct.
        let window_start = pos1.saturating_sub(half_w);
        let window_end = (pos1 + half_w + 1).min(num_items);
        candidates.clear();
        weights.clear();
        for &(cand_item, cand_rating) in &sorted_pool[window_start..window_end] {
            if cand_item == item1 {
                continue;
            }
            candidates.push(cand_item);
            weights.push(window_info_gain(
                item1_rating, cand_rating, item1, cand_item, current_stds, matchmaking_sharpness,
            ));
        }
        // For num_items >= 2 the window around item1 always holds at least one
        // other item, so a distinct opponent always exists.
        debug_assert!(!candidates.is_empty(), "top-heavy opponent window was empty");

        let total_candidate_weight: f64 = weights.iter().sum();
        let selected = if total_candidate_weight == 0.0 {
            rng.random_range(0..candidates.len())
        } else {
            weighted_random_select(&weights, total_candidate_weight, rng)
        };
        let item2 = candidates[selected];

        let p = position_probability(
            local_first_counts[item1], local_edge_counts[item1],
            local_first_counts[item2], local_edge_counts[item2],
        );
        if rng.random::<f64>() < p {
            pairs.push((item1, item2));
            local_first_counts[item1] += 1;
        } else {
            pairs.push((item2, item1));
            local_first_counts[item2] += 1;
        }
        local_edge_counts[item1] += 1;
        local_edge_counts[item2] += 1;
    }

    pairs
}

// ---------------------------------------------------------------------------
// Lineup generation
// ---------------------------------------------------------------------------
//
// A lineup judgement shows the judge `lineup_size` items at once. Lineup
// selection mirrors pair selection stage-for-stage — random, then
// nearest-neighbour, then info-gain matchmaking — but places items into
// `lineup_size`-item lineups. No position orientation is assigned here: the
// caller shuffles each lineup into presentation slots, and the scoring engine
// estimates the per-slot bias from the slots recorded on the folded edges.

/// Reject a lineup size outside the supported range at the point of request.
///
/// `lineup::winner_dist_to_edges` rejects the same range when folding a
/// judgement, but that is only reached after the judge has been called. Failing
/// here means a caller learns the size is unsupported before spending anything
/// on it.
pub(crate) fn assert_lineup_size(lineup_size: usize) {
    assert!(
        (MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE).contains(&lineup_size),
        "lineup_size must be between {MIN_LINEUP_SIZE} and {MAX_LINEUP_SIZE}, got {lineup_size}"
    );
}

/// Generate any requested number of cumulatively balanced uniform lineups.
/// Matchmaking starts random, then becomes rating-adjacent and finally
/// information-gain weighted as the minimum incident-edge count accumulates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_uniform_lineups_indexed(
    num_items: usize,
    lineups_count: usize,
    lineup_size: usize,
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    sharpness: f64,
    edge_counts: &[usize],
    rng: &mut StdRng,
) -> Vec<IndexedLineup> {
    if num_items < lineup_size {
        return Vec::new();
    }
    let mut lineups = Vec::with_capacity(lineups_count);
    let mut local_edge_counts = edge_counts.to_vec();
    let judgements_to_include_every_item_once = num_items.div_ceil(lineup_size);

    while lineups.len() < lineups_count {
        let iteration_size =
            (lineups_count - lineups.len()).min(judgements_to_include_every_item_once);
        let min_edge_count = local_edge_counts.iter().copied().min().unwrap_or(0);
        let slots = iteration_size.saturating_mul(lineup_size).min(num_items);
        let pool = least_edge_count_pool(&local_edge_counts, slots, rng);

        let mut iteration_lineups = if min_edge_count == 0 {
            random_lineups(&pool, iteration_size, lineup_size, rng)
        } else if min_edge_count == 1 {
            nearest_neighbour_lineups(&pool, current_ratings, iteration_size, lineup_size, rng)
        } else {
            info_gain_lineups(
                &pool,
                current_ratings,
                current_stds,
                sharpness,
                iteration_size,
                lineup_size,
                rng,
            )
        };

        // When the item count is not divisible by the lineup size, complete the
        // final lineup with the unselected items plus distinct opponents that
        // were already selected during this iteration.
        if iteration_lineups.len() < iteration_size {
            let mut seen = vec![false; num_items];
            for &item in iteration_lineups.iter().flatten() {
                seen[item] = true;
            }
            let mut final_lineup: Vec<usize> = pool
                .iter()
                .copied()
                .filter(|&item| !seen[item])
                .collect();
            let anchor = final_lineup[0];
            let mut candidates: Vec<usize> = pool
                .iter()
                .copied()
                .filter(|item| !final_lineup.contains(item))
                .collect();
            candidates.shuffle(rng);

            while final_lineup.len() < lineup_size {
                let selected = if min_edge_count == 0 {
                    0
                } else if min_edge_count == 1 {
                    candidates
                        .iter()
                        .enumerate()
                        .min_by(|a, b| {
                            let item_a = *a.1;
                            let item_b = *b.1;
                            (current_ratings[anchor] - current_ratings[item_a])
                                .abs()
                                .partial_cmp(
                                    &(current_ratings[anchor] - current_ratings[item_b]).abs(),
                                )
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .expect("lineup completion requires another item")
                } else {
                    let weights: Vec<f64> = candidates
                        .iter()
                        .map(|&item| {
                            window_info_gain(
                                current_ratings[anchor],
                                current_ratings[item],
                                anchor,
                                item,
                                current_stds,
                                sharpness,
                            )
                        })
                        .collect();
                    let total_weight: f64 = weights.iter().sum();
                    if total_weight == 0.0 {
                        rng.random_range(0..candidates.len())
                    } else {
                        weighted_random_select(&weights, total_weight, rng)
                    }
                };
                final_lineup.push(candidates.swap_remove(selected));
            }
            iteration_lineups.push(final_lineup);
        }

        assert!(!iteration_lineups.is_empty(), "uniform lineup generation made no progress");
        // Each member of a full winner distribution participates in one
        // derived edge against every other lineup member. These counts are a
        // local projection used only while planning this call; actual surviving
        // edges are recorded from the returned judgement afterward.
        let projected_edges_per_item = lineup_size - 1;
        for &item in iteration_lineups.iter().flatten() {
            local_edge_counts[item] += projected_edges_per_item;
        }
        lineups.extend(iteration_lineups);
    }

    lineups
}

/// While some items have no edges, place the selected items randomly.
fn random_lineups(
    item_pool: &[usize],
    max_lineups: usize,
    lineup_size: usize,
    rng: &mut impl Rng,
) -> Vec<IndexedLineup> {
    let mut order = item_pool.to_vec();
    order.shuffle(rng);
    order
        .chunks_exact(lineup_size)
        .take(max_lineups)
        .map(|c| c.to_vec())
        .collect()
}

/// Once every item has an edge, place adjacent rating-neighbours into lineups.
/// Shuffle before the stable sort so equal-rated items order randomly.
fn nearest_neighbour_lineups(
    item_pool: &[usize],
    current_ratings: &[f64],
    max_lineups: usize,
    lineup_size: usize,
    rng: &mut impl Rng,
) -> Vec<IndexedLineup> {
    let mut pool: Vec<(usize, f64)> = item_pool
        .iter()
        .map(|&i| (i, current_ratings[i]))
        .collect();
    pool.shuffle(rng);
    pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    pool.chunks_exact(lineup_size)
        .take(max_lineups)
        .map(|c| c.iter().map(|&(item, _)| item).collect())
        .collect()
}

/// Once every item has at least two edges, build info-gain lineups. Pick a random
/// unpaired item1, then draw `lineup_size - 1` distinct opponents from the rating
/// window around it, each weighted by info gain so closely-matched (more
/// informative) lineups are favoured. All of them are removed once placed.
/// Mirrors `info_gain_pairs` with repeated opponent draws.
#[allow(clippy::too_many_arguments)]
fn info_gain_lineups(
    item_pool: &[usize],
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    sharpness: f64,
    max_lineups: usize,
    lineup_size: usize,
    rng: &mut impl Rng,
) -> Vec<IndexedLineup> {
    let mut sorted_pool: Vec<(usize, f64)> = item_pool
        .iter()
        .map(|&i| (i, current_ratings[i]))
        .collect();
    sorted_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_pool.len();
    let mut live_positions: Vec<usize> = (0..n).collect();
    let mut live_idx_of: Vec<Option<usize>> = (0..n).map(Some).collect();
    let (mut previous_live, mut next_live) = make_live_neighbour_links(n);
    let mut candidates: Vec<usize> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + lineup_size);
    let mut weights: Vec<f64> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + lineup_size);

    let opponents_needed = lineup_size - 1;
    let mut lineups: Vec<IndexedLineup> = Vec::with_capacity(max_lineups);

    for _ in 0..max_lineups {
        if live_positions.len() < lineup_size {
            break;
        }

        // item1: random live entry.
        let live_idx1 = rng.random_range(0..live_positions.len());
        let (pos1, left, right) = remove_live_position(
            &mut live_positions,
            &mut live_idx_of,
            &mut previous_live,
            &mut next_live,
            live_idx1,
        );
        let (item1, item1_rating) = sorted_pool[pos1];

        // Gather every opponent before selecting any of them. The ordinary
        // fixed window is unchanged; only an exhausted window expands far
        // enough through live rating neighbours to make the lineup possible.
        collect_live_window_candidates(
            pos1,
            left,
            right,
            &previous_live,
            &next_live,
            n,
            opponents_needed,
            &mut candidates,
        );
        weights.clear();
        for &pos in &candidates {
            weights.push(window_info_gain(
                item1_rating,
                sorted_pool[pos].1,
                item1,
                sorted_pool[pos].0,
                current_stds,
                sharpness,
            ));
        }

        let mut lineup: IndexedLineup = Vec::with_capacity(lineup_size);
        lineup.push(item1);
        for _ in 0..opponents_needed {
            let total_weight: f64 = weights.iter().sum();
            let selected = if total_weight == 0.0 {
                rng.random_range(0..candidates.len())
            } else {
                weighted_random_select(&weights, total_weight, rng)
            };
            let pos = candidates.remove(selected);
            weights.remove(selected);
            let live_idx = live_idx_of[pos].expect("candidate must be alive");
            remove_live_position(
                &mut live_positions,
                &mut live_idx_of,
                &mut previous_live,
                &mut next_live,
                live_idx,
            );
            lineup.push(sorted_pool[pos].0);
        }
        lineups.push(lineup);
    }

    lineups
}

/// Generate top-heavy lineups. item1 is sampled from the selection weights
/// (concentrating on contenders); the remaining `lineup_size - 1` items are
/// distinct info-gain opponents from the rating window around item1. Mirrors
/// `generate_top_heavy_pairings_indexed` with repeated opponent draws.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_top_heavy_lineups_indexed(
    num_items: usize,
    lineups_count: usize,
    lineup_size: usize,
    selection_weights: &[f64],
    current_ratings: &[f64],
    current_stds: Option<&[f64]>,
    matchmaking_sharpness: f64,
    rng: &mut StdRng,
) -> Vec<IndexedLineup> {
    if num_items < lineup_size {
        return Vec::new();
    }
    assert_eq!(selection_weights.len(), num_items, "selection_weights length mismatch");
    assert_eq!(current_ratings.len(), num_items, "current_ratings length mismatch");
    if let Some(stds) = current_stds {
        assert_eq!(stds.len(), num_items, "current_stds length mismatch");
    }

    let total_weight: f64 = selection_weights.iter().sum();
    assert!(
        total_weight > 0.0,
        "top-heavy lineup selection requires positive total item-selection weight"
    );

    let mut sorted_pool: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, current_ratings[i]))
        .collect();
    sorted_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_position = vec![0usize; num_items];
    for (pos, &(item, _)) in sorted_pool.iter().enumerate() {
        sorted_position[item] = pos;
    }

    let half_w = OPPONENT_WINDOW_SIZE / 2;
    let n = num_items;
    let opponents_needed = lineup_size - 1;

    let mut lineups: Vec<IndexedLineup> = Vec::with_capacity(lineups_count);

    for _ in 0..lineups_count {
        let item1 = weighted_random_select(selection_weights, total_weight, rng);
        let item1_rating = current_ratings[item1];
        let pos1 = sorted_position[item1];

        // The rest of the lineup, drawn from the window around item1. `chosen`
        // marks item1 and any already-picked opponent as unavailable within
        // this lineup (sampling item1 is with-replacement across lineups, so no
        // global alive array — exclusion is per-lineup).
        let mut chosen = vec![false; n];
        chosen[pos1] = true;
        let mut lineup: IndexedLineup = Vec::with_capacity(lineup_size);
        lineup.push(item1);
        for _ in 0..opponents_needed {
            // The fixed window normally holds every opponent the lineup needs.
            // Widen it only when it does not — with few items, or once earlier
            // draws have consumed the neighbourhood.
            let mut window_radius = half_w;
            let (mut candidates, mut weights) = (Vec::new(), Vec::new());
            loop {
                let window_start = pos1.saturating_sub(window_radius);
                let window_end = (pos1 + window_radius + 1).min(n);
                candidates.clear();
                weights.clear();
                for p in window_start..window_end {
                    if !chosen[p] {
                        candidates.push(p);
                        weights.push(window_info_gain(
                            item1_rating,
                            sorted_pool[p].1,
                            item1,
                            sorted_pool[p].0,
                            current_stds,
                            matchmaking_sharpness,
                        ));
                    }
                }
                if !candidates.is_empty() || (window_start == 0 && window_end == n) {
                    break;
                }
                window_radius *= 2;
            }
            // num_items >= lineup_size guarantees an unchosen item exists.
            debug_assert!(!candidates.is_empty(), "top-heavy lineup window was empty");
            let total_candidate_weight: f64 = weights.iter().sum();
            let selected = if total_candidate_weight == 0.0 {
                rng.random_range(0..candidates.len())
            } else {
                weighted_random_select(&weights, total_candidate_weight, rng)
            };
            let pos = candidates[selected];
            chosen[pos] = true;
            lineup.push(sorted_pool[pos].0);
        }
        lineups.push(lineup);
    }

    lineups
}

/// Sample an index proportionally to `weights` (non-negative, summing to
/// `total_weight > 0`). Scale-invariant: `r` and the weights shrink together,
/// so it behaves identically whether the weights sum to 1e-40 or 1e+40 — no
/// absolute epsilon, which would misfire once the total dropped below it.
/// Zero-weight entries are skipped outright, so they can never be selected;
/// if floating-point residue leaves `r` a hair above zero after the loop,
/// fall back to the last positive-weight index.
fn weighted_random_select(weights: &[f64], total_weight: f64, rng: &mut impl Rng) -> usize {
    let mut r = rng.random::<f64>() * total_weight;
    let mut last_positive = 0;
    for (j, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        last_positive = j;
        r -= w;
        if r <= 0.0 {
            return j;
        }
    }
    last_positive
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE};

    #[test]
    fn test_info_gain_equal_ratings() {
        let gain = calculate_info_gain(1.0, 1.0, 1.0);
        assert!((gain - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_info_gain_unequal_ratings() {
        let gain = calculate_info_gain(100.0, 1.0, 1.0);
        assert!(gain < 0.01);
    }

    #[test]
    fn test_info_gain_sharpness() {
        let gain_low = calculate_info_gain(1.0, 2.0, 0.5);
        let gain_high = calculate_info_gain(1.0, 2.0, 2.0);
        assert!(gain_high < gain_low);
    }

    #[test]
    fn test_integrated_info_gain_zero_std_matches_plugin_exactly() {
        // kappa is exactly 1.0 when both stds are 0, so the two functions must
        // agree bit-for-bit — this is what makes the integrated gain a strict
        // generalization of the plug-in gain.
        for &(a, b, s) in &[(1.0, 1.0, 1.0), (0.3, -1.2, 1.0), (2.0, 0.5, 2.0), (5.0, -5.0, 0.5)] {
            assert_eq!(
                calculate_info_gain(a, b, s),
                calculate_integrated_info_gain(a, 0.0, b, 0.0, s),
            );
        }
    }

    #[test]
    fn test_integrated_info_gain_uncertainty_raises_gain() {
        // For a fixed rating gap, integration can only pull p toward 0.5, so
        // gain rises monotonically with uncertainty and stays below the 0.25 cap.
        let g0 = calculate_integrated_info_gain(2.0, 0.0, 0.0, 0.0, 1.0);
        let g1 = calculate_integrated_info_gain(2.0, 1.0, 0.0, 0.5, 1.0);
        let g2 = calculate_integrated_info_gain(2.0, 3.0, 0.0, 3.0, 1.0);
        assert!(g0 < g1 && g1 < g2, "gain must rise with uncertainty: {g0} {g1} {g2}");
        assert!(g2 <= 0.25);
    }

    #[test]
    fn test_integrated_info_gain_symmetric() {
        let g_ab = calculate_integrated_info_gain(1.7, 0.4, -0.3, 1.1, 1.3);
        let g_ba = calculate_integrated_info_gain(-0.3, 1.1, 1.7, 0.4, 1.3);
        assert!((g_ab - g_ba).abs() < 1e-15);
    }

    #[test]
    fn test_integrated_info_gain_large_uncertainty_flattens() {
        // With stds much larger than the gaps, near and far matchups score
        // almost the same, while the plug-in gains differ by an order of
        // magnitude — the exploration regime the integration is meant to add.
        let close = calculate_integrated_info_gain(0.0, 5.0, 0.0, 5.0, 1.0);
        let far = calculate_integrated_info_gain(4.0, 5.0, 0.0, 5.0, 1.0);
        assert!(close / far < 1.5, "integrated gains should be near-flat: {close} vs {far}");
        let plug_far = calculate_info_gain(4.0, 0.0, 1.0);
        assert!(0.25 / plug_far > 10.0, "plug-in should still be strongly peaked");
    }

    #[test]
    fn test_effective_judgement_distribution_uniform_user_choice() {
        let edges = vec![10, 10];
        let result = get_effective_judgement_distribution(JudgementDistribution::Uniform, 2, &edges, 3);
        assert_eq!(result, JudgementDistribution::Uniform);
    }

    #[test]
    fn test_effective_judgement_distribution_uniform_stage() {
        let edges = vec![1, 10]; // Item 0 below minimum
        let result = get_effective_judgement_distribution(JudgementDistribution::TopHeavy, 2, &edges, 3);
        assert_eq!(result, JudgementDistribution::Uniform);
    }

    #[test]
    fn test_effective_judgement_distribution_main_phase() {
        let edges = vec![10, 10];
        let result = get_effective_judgement_distribution(JudgementDistribution::TopHeavy, 2, &edges, 3);
        assert_eq!(result, JudgementDistribution::TopHeavy);
    }

    #[test]
    fn test_uniform_pairings_include_every_item() {
        let item_ids: Vec<i64> = (100..111).collect();
        let ratings = vec![1.0; item_ids.len()];
        let zeros = vec![0usize; item_ids.len()];
        let pairs = generate_uniform_pairings(&item_ids, 6, &ratings, None, 1.0, &zeros, &zeros);

        assert_eq!(pairs.len(), 6);

        let mut seen = std::collections::HashSet::new();
        for (a, b) in &pairs {
            assert!(item_ids.contains(a), "ID {a} not in item_ids");
            assert!(item_ids.contains(b), "ID {b} not in item_ids");
            assert_ne!(a, b);
            seen.insert(*a);
            seen.insert(*b);
        }
        assert_eq!(seen.len(), item_ids.len());
    }

    #[test]
    fn test_uniform_info_gain_pairs_fill_large_passes() {
        let num_items = 500;
        let ratings: Vec<f64> = (0..num_items).map(|i| i as f64 * 0.01).collect();

        for seed in 0..128 {
            let mut rng = make_rng(Some(seed), crate::seed::SUBSYSTEM_PAIRING);
            let item_pool: Vec<usize> = (0..num_items).collect();
            let pairs = info_gain_pairs(
                &item_pool,
                &ratings,
                None,
                1.0,
                num_items / 2,
                &mut rng,
            );
            assert_eq!(
                pairs.len(),
                num_items / 2,
                "short pair pass for seed {seed}"
            );

            let mut seen = vec![false; num_items];
            for (a, b) in pairs {
                assert!(!seen[a], "item {a} was paired twice for seed {seed}");
                assert!(!seen[b], "item {b} was paired twice for seed {seed}");
                seen[a] = true;
                seen[b] = true;
            }
        }
    }

    #[test]
    fn test_live_window_expands_only_after_fixed_window_is_exhausted() {
        let num_items = 201;

        // With a dense pool, position 100 sees exactly the existing fixed
        // window: positions 50..=150 except for the removed anchor itself.
        let mut live_positions: Vec<usize> = (0..num_items).collect();
        let mut live_idx_of: Vec<Option<usize>> = (0..num_items).map(Some).collect();
        let (mut previous_live, mut next_live) = make_live_neighbour_links(num_items);
        let anchor_idx = live_idx_of[100].unwrap();
        let (_, left, right) = remove_live_position(
            &mut live_positions,
            &mut live_idx_of,
            &mut previous_live,
            &mut next_live,
            anchor_idx,
        );
        let mut candidates = Vec::new();
        collect_live_window_candidates(
            100,
            left,
            right,
            &previous_live,
            &next_live,
            num_items,
            2,
            &mut candidates,
        );
        assert_eq!(candidates.len(), OPPONENT_WINDOW_SIZE);
        assert_eq!(candidates.first(), Some(&50));
        assert_eq!(candidates.last(), Some(&150));

        // Leave only the anchor and two entries beyond that window. The same
        // lookup now follows the live links outward just far enough to find both.
        let mut live_positions: Vec<usize> = (0..num_items).collect();
        let mut live_idx_of: Vec<Option<usize>> = (0..num_items).map(Some).collect();
        let (mut previous_live, mut next_live) = make_live_neighbour_links(num_items);
        for pos in 0..num_items {
            if pos == 0 || pos == 100 || pos == 200 {
                continue;
            }
            let live_idx = live_idx_of[pos].unwrap();
            remove_live_position(
                &mut live_positions,
                &mut live_idx_of,
                &mut previous_live,
                &mut next_live,
                live_idx,
            );
        }
        let anchor_idx = live_idx_of[100].unwrap();
        let (_, left, right) = remove_live_position(
            &mut live_positions,
            &mut live_idx_of,
            &mut previous_live,
            &mut next_live,
            anchor_idx,
        );
        collect_live_window_candidates(
            100,
            left,
            right,
            &previous_live,
            &next_live,
            num_items,
            2,
            &mut candidates,
        );
        assert_eq!(candidates, vec![0, 200]);
    }

    #[test]
    fn test_pairs_with_one_edge_per_item_break_rating_ties_randomly() {
        // Once every item has exactly one edge, the selector sorts by rating and
        // pairs adjacent neighbours. When ratings are all equal — the binary
        // case where every winner ties and every loser ties — pairing among the
        // tied items must be random, not fixed index order.
        let num_items = 8;
        let ratings = vec![1.0; num_items]; // all tied
        let first_counts = vec![0usize; num_items];
        let edges = vec![1usize; num_items];

        // Unordered partnerships (sorted pair) for a given seed, so we test who
        // is matched with whom, independent of the separate orientation flip.
        let partnerships = |s: u64| -> Vec<(usize, usize)> {
            let mut rng = make_rng(Some(s), crate::seed::SUBSYSTEM_PAIRING);
            let pairs = generate_uniform_pairings_indexed(
                num_items, num_items / 2, &ratings, None, 1.0, &first_counts, &edges, &mut rng,
            );
            let mut ps: Vec<(usize, usize)> = pairs
                .into_iter()
                .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
                .collect();
            ps.sort();
            ps
        };

        // The index-order fallback a plain (non-shuffled) sort would produce.
        let index_order: Vec<(usize, usize)> = (0..num_items / 2).map(|i| (2 * i, 2 * i + 1)).collect();

        let seeds: Vec<u64> = (0..25).collect();
        let results: Vec<Vec<(usize, usize)>> = seeds.iter().map(|&s| partnerships(s)).collect();

        // (1) Varies across seeds => not deterministic.
        let distinct: std::collections::HashSet<_> = results.iter().cloned().collect();
        assert!(distinct.len() > 1, "tied-item pairing did not vary across seeds — it is deterministic");

        // (2) Not the fixed index-order pairing (guards the plain-sort regression).
        assert!(
            results.iter().any(|r| *r != index_order),
            "tied-item pairing always matched the index-order fallback"
        );
    }

    #[test]
    fn test_public_uniform_pairings_use_ratings_after_uniform_stage() {
        // With every item at 2+ edges the public wrapper must run info-gain
        // matchmaking on the supplied ratings, pairing rating-neighbours.
        // Regression test for the wrapper hardcoding edge_counts to zeros,
        // which forced the initial random branch and made `current_ratings`
        // and `sharpness` dead parameters.
        let item_ids: Vec<i64> = (0..10).collect();
        let ratings: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();
        let first = vec![1usize; 10];
        let edges = vec![2usize; 10];

        let mut total_gap = 0.0;
        let mut count = 0usize;
        for _ in 0..200 {
            let pairs = generate_uniform_pairings(&item_ids, 5, &ratings, None, 1.0, &first, &edges);
            for (a, b) in pairs {
                total_gap += (ratings[a as usize] - ratings[b as usize]).abs();
                count += 1;
            }
        }
        let mean_gap = total_gap / count as f64;
        // Random pairing over ratings 0,2,…,18 averages a gap of ~7.3;
        // info-gain matchmaking pulls it well below that.
        assert!(mean_gap < 5.0, "expected rating-local pairs, mean gap was {mean_gap}");
    }

    #[test]
    fn test_position_probability_no_history() {
        // With zero counts on both sides, smoothing gives both items a ratio of
        // 1/2, so the formula must collapse to a fair coin flip.
        let p = position_probability(0, 0, 0, 0);
        assert!((p - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_position_probability_equal_ratios() {
        // Equal ratios at any sample size must give 0.5.
        assert!((position_probability(5, 10, 5, 10) - 0.5).abs() < 1e-12);
        assert!((position_probability(50, 100, 500, 1000) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_position_probability_skewed_pulls_toward_balance() {
        // A has gone first 70/100 times, B 30/100. A should be down-weighted.
        let p = position_probability(70, 100, 30, 100);
        assert!(p < 0.5, "A should be less likely to go first; got {}", p);
        // ratio_A = 71/102, ratio_B = 31/102 → P(A) = 31 / (71 + 31).
        let expected = 31.0 / 102.0;
        let denom = 71.0 / 102.0 + 31.0 / 102.0;
        assert!((p - expected / denom).abs() < 1e-12);
    }

    #[test]
    fn test_position_probability_smoothing_damps_small_samples() {
        // A has accumulated 1 edge and went first. Without smoothing this would be
        // ratio 1.0; with Laplace smoothing it is 2/3, far from extreme.
        let p = position_probability(1, 1, 0, 0);
        // ratio_A = 2/3, ratio_B = 1/2 → P(A) = (1/2) / (2/3 + 1/2) = 3/7.
        assert!((p - 3.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_position_probability_exact_at_various_scales() {
        // The formula is fully deterministic. Each expected value below is
        // computed by hand from the smoothed-ratio definition.

        // 7/10 vs 3/10:        ratios 8/12, 4/12.   P = 4/12 / 12/12 = 1/3.
        let p_tiny = position_probability(7, 10, 3, 10);
        assert!((p_tiny - 1.0 / 3.0).abs() < 1e-12);

        // 70/100 vs 30/100:    ratios 71/102, 31/102.   P = 31/102.
        let p_mid = position_probability(70, 100, 30, 100);
        assert!((p_mid - 31.0 / 102.0).abs() < 1e-12);

        // 700/1000 vs 300/1000:    ratios 701/1002, 301/1002.   P = 301/1002.
        let p_large = position_probability(700, 1000, 300, 1000);
        assert!((p_large - 301.0 / 1002.0).abs() < 1e-12);

        // As N grows the smoothing's effect shrinks, so the probability moves
        // toward the un-smoothed ratio (3/10 = 0.3). Each larger sample is
        // strictly closer to that limit than the smaller one before it.
        let limit = 0.3;
        assert!((p_large - limit).abs() < (p_mid - limit).abs());
        assert!((p_mid - limit).abs() < (p_tiny - limit).abs());
    }

    #[test]
    fn test_weighted_random_select_tiny_weights_follow_ratios() {
        // Underflow-scale weights — as produced when every selection area is
        // normal_cdf of a hugely negative z — must still be sampled by their
        // ratios. Regression test for an absolute epsilon (r < 1e-10) that made
        // any total below it always return index 0, even a zero-weight item.
        let weights = [0.0, 1e-45, 3e-45];
        let total: f64 = weights.iter().sum();
        let mut rng = make_rng(Some(3), crate::seed::SUBSYSTEM_PAIRING);

        let mut counts = [0usize; 3];
        for _ in 0..2000 {
            counts[weighted_random_select(&weights, total, &mut rng)] += 1;
        }
        assert_eq!(counts[0], 0, "zero-weight item must never be selected: {counts:?}");
        assert!(counts[1] > 0 && counts[2] > 0, "both positive-weight items should appear: {counts:?}");
        assert!(counts[2] > counts[1], "3:1 weight ratio should show in the counts: {counts:?}");
    }

    #[test]
    fn test_top_heavy_pairings() {
        let item_ids: Vec<i64> = (0..10).collect();
        let selection_weights: Vec<f64> = (0..10).map(|i| if i < 3 { 0.8 } else { 0.05 }).collect();
        let ratings: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let zeros = vec![0usize; 10];

        let pairs = generate_top_heavy_pairings(&item_ids, 5, &selection_weights, &ratings, None, 1.0, &zeros, &zeros);
        assert_eq!(pairs.len(), 5);
        for (a, b) in &pairs {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn test_top_heavy_item1_from_weights_concentrates_on_contenders() {
        // Items 0,1,2 carry essentially all the selection weight and share a high
        // rating; 3-9 carry ~none and sit far below. item1 is always a contender
        // (drawn from the weights), and its info-gain opponent is the nearest in
        // rating — another contender — so the heavy items vastly out-appear the
        // tail across a long generated sequence.
        let item_ids: Vec<i64> = (0..10).collect();
        let selection_weights: Vec<f64> = (0..10).map(|i| if i < 3 { 1.0 } else { 0.0001 }).collect();
        let ratings: Vec<f64> = (0..10).map(|i| if i < 3 { 5.0 } else { 0.0 }).collect();
        let zeros = vec![0usize; 10];

        let pairs = generate_top_heavy_pairings(&item_ids, 200, &selection_weights, &ratings, None, 1.0, &zeros, &zeros);
        let mut appearances = [0usize; 10];
        for (a, b) in &pairs {
            appearances[*a as usize] += 1;
            appearances[*b as usize] += 1;
        }
        let top_3: usize = appearances[0] + appearances[1] + appearances[2];
        let tail: usize = appearances[3..].iter().sum();
        assert!(top_3 > tail, "contenders ({top_3}) should out-appear tail ({tail})");
    }

    #[test]
    fn test_top_heavy_opponent_is_rating_local() {
        // With equal selection weights (item1 uniform) and spread-out ratings,
        // the info-gain opponent picks the nearest-rated items far more often than
        // a random opponent would, so the mean rating gap stays small.
        let item_ids: Vec<i64> = (0..10).collect();
        let selection_weights = vec![1.0; 10];
        let ratings: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let zeros = vec![0usize; 10];

        let pairs = generate_top_heavy_pairings(&item_ids, 2000, &selection_weights, &ratings, None, 1.0, &zeros, &zeros);
        let mut total_gap = 0.0;
        for (a, b) in &pairs {
            total_gap += (ratings[*a as usize] - ratings[*b as usize]).abs();
        }
        let mean_gap = total_gap / pairs.len() as f64;
        // A uniformly-random opponent over ratings 0..9 averages a gap of ~3.3;
        // info-gain matchmaking pulls this well below that.
        assert!(mean_gap < 2.5, "info-gain opponents should be rating-local; mean gap was {mean_gap}");
    }

    #[test]
    fn test_top_heavy_opponent_window_flattens_with_uncertainty() {
        // Same setup as test_top_heavy_opponent_is_rating_local, but with
        // posterior stds much larger than the rating gaps the integrated gain
        // flattens the window, so the mean rating gap grows toward the
        // random-opponent value instead of staying tight.
        let item_ids: Vec<i64> = (0..10).collect();
        let selection_weights = vec![1.0; 10];
        let ratings: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let stds = vec![10.0; 10];
        let zeros = vec![0usize; 10];

        let mean_gap = |stds: Option<&[f64]>| -> f64 {
            let pairs = generate_top_heavy_pairings(
                &item_ids, 2000, &selection_weights, &ratings, stds, 1.0, &zeros, &zeros,
            );
            let total: f64 = pairs.iter()
                .map(|(a, b)| (ratings[*a as usize] - ratings[*b as usize]).abs())
                .sum();
            total / pairs.len() as f64
        };

        let gap_plugin = mean_gap(None);
        let gap_integrated = mean_gap(Some(&stds));
        assert!(
            gap_integrated > gap_plugin + 0.3,
            "huge stds should flatten opponent selection: plug-in gap {gap_plugin}, integrated gap {gap_integrated}"
        );
    }

    // --- Lineup generation tests ---

    /// Every lineup holds `lineup_size` distinct members.
    fn assert_lineups_distinct(lineups: &[IndexedLineup], lineup_size: usize) {
        for lineup in lineups {
            assert_eq!(lineup.len(), lineup_size, "lineup {lineup:?} has the wrong size");
            let mut sorted = lineup.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), lineup_size, "lineup {lineup:?} has a repeat");
        }
    }

    #[test]
    fn test_initial_uniform_lineups_random_distinct() {
        let ratings = vec![1.0; 12];
        let edges = vec![0usize; 12];
        let mut rng = make_rng(Some(1), crate::seed::SUBSYSTEM_PAIRING);
        let lineups = generate_uniform_lineups_indexed(12, 4, 3, &ratings, None, 1.0, &edges, &mut rng);
        assert_eq!(lineups.len(), 4);
        assert_lineups_distinct(&lineups, 3);
    }

    #[test]
    fn test_uniform_lineups_info_gain_stage_distinct() {
        let ratings: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let edges = vec![2usize; 12]; // minimum reached: info-gain stage
        let mut rng = make_rng(Some(2), crate::seed::SUBSYSTEM_PAIRING);
        let lineups = generate_uniform_lineups_indexed(12, 4, 3, &ratings, None, 1.0, &edges, &mut rng);
        assert_eq!(lineups.len(), 4);
        assert_lineups_distinct(&lineups, 3);
    }

    #[test]
    fn test_planned_lineups_project_lineup_size_minus_one_edges_per_item() {
        let ratings: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let no_edges = vec![0usize; 12];
        let two_edges_each = vec![2usize; 12];

        // Planning eight size-three lineups in one call should make the same
        // stage transition as planning four, recording their two incident edges
        // per item, then planning four more. A planned lineup appearance must
        // not be added to an edge counter as though it represented only one
        // edge.
        let mut combined_rng = make_rng(Some(29), crate::seed::SUBSYSTEM_PAIRING);
        let combined = generate_uniform_lineups_indexed(
            12,
            8,
            3,
            &ratings,
            None,
            1.0,
            &no_edges,
            &mut combined_rng,
        );

        let mut split_rng = make_rng(Some(29), crate::seed::SUBSYSTEM_PAIRING);
        let mut split = generate_uniform_lineups_indexed(
            12,
            4,
            3,
            &ratings,
            None,
            1.0,
            &no_edges,
            &mut split_rng,
        );
        split.extend(generate_uniform_lineups_indexed(
            12,
            4,
            3,
            &ratings,
            None,
            1.0,
            &two_edges_each,
            &mut split_rng,
        ));

        assert_eq!(combined, split);
    }

    /// At each selection stage and supported lineup size, requesting enough
    /// lineups for every item to appear once returns correctly-sized, internally
    /// distinct lineups containing every item.
    #[test]
    fn test_uniform_lineups_every_size() {
        let num_items = 60;
        for lineup_size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            for (stage, edge_count) in [("random", 0usize), ("neighbour", 1), ("info-gain", 2)] {
                let ratings: Vec<f64> = (0..num_items).map(|i| i as f64 * 0.1).collect();
                let edges = vec![edge_count; num_items];
                let mut rng = make_rng(Some(7), crate::seed::SUBSYSTEM_PAIRING);
                let wanted = num_items.div_ceil(lineup_size);
                let lineups = generate_uniform_lineups_indexed(
                    num_items, wanted, lineup_size, &ratings, None, 1.0, &edges, &mut rng,
                );
                assert_eq!(
                    lineups.len(), wanted,
                    "size {lineup_size} stage {stage}: too few lineups"
                );
                assert_lineups_distinct(&lineups, lineup_size);

                let mut counts = vec![0usize; num_items];
                for item in lineups.iter().flatten() {
                    counts[*item] += 1;
                }
                assert!(
                    counts.iter().all(|&count| count >= 1),
                    "size {lineup_size} stage {stage}: an item is missing {counts:?}"
                );
                assert_eq!(
                    counts.iter().map(|&count| count - 1).sum::<usize>(),
                    wanted * lineup_size - num_items,
                    "size {lineup_size} stage {stage}: unexpected overlap"
                );
            }
        }
    }

    #[test]
    fn test_uniform_info_gain_lineups_fill_large_passes() {
        let num_items = 500;
        let ratings: Vec<f64> = (0..num_items).map(|i| i as f64 * 0.01).collect();

        for seed in 0..128 {
            let mut rng = make_rng(Some(seed), crate::seed::SUBSYSTEM_PAIRING);
            let item_pool: Vec<usize> = (0..num_items).collect();
            let lineups = info_gain_lineups(
                &item_pool,
                &ratings,
                None,
                1.0,
                num_items / 3,
                3,
                &mut rng,
            );
            assert_eq!(
                lineups.len(),
                num_items / 3,
                "short lineup pass for seed {seed}"
            );

            let mut seen = vec![false; num_items];
            for item in lineups.iter().flatten() {
                assert!(!seen[*item], "item {item} appeared in two lineups for seed {seed}");
                seen[*item] = true;
            }
        }
    }

    #[test]
    fn test_top_heavy_lineups_distinct_and_sized() {
        let selection_weights: Vec<f64> = (0..10).map(|i| if i < 3 { 0.8 } else { 0.05 }).collect();
        let ratings: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut rng = make_rng(Some(3), crate::seed::SUBSYSTEM_PAIRING);
        let lineups = generate_top_heavy_lineups_indexed(10, 5, 3, &selection_weights, &ratings, None, 1.0, &mut rng);
        assert_eq!(lineups.len(), 5);
        assert_lineups_distinct(&lineups, 3);
    }

    /// Top-heavy lineups stay correctly sized and distinct at every supported
    /// size, including when the item count barely covers one lineup.
    #[test]
    fn test_top_heavy_lineups_every_size() {
        for lineup_size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            for num_items in [lineup_size, lineup_size + 1, 40] {
                let selection_weights = vec![1.0; num_items];
                let ratings: Vec<f64> = (0..num_items).map(|i| i as f64).collect();
                let mut rng = make_rng(Some(11), crate::seed::SUBSYSTEM_PAIRING);
                let lineups = generate_top_heavy_lineups_indexed(
                    num_items, 20, lineup_size, &selection_weights, &ratings, None, 1.0, &mut rng,
                );
                assert_eq!(
                    lineups.len(), 20,
                    "size {lineup_size}, {num_items} items: wrong lineup count"
                );
                assert_lineups_distinct(&lineups, lineup_size);
            }
        }
    }

    #[test]
    fn test_top_heavy_lineups_concentrate_on_contenders() {
        // Items 0,1,2 hold the weight and share a high rating; the rest are near
        // zero. Every lineup's item1 is a contender, and its info-gain opponents
        // are the nearest in rating (also contenders), so the top three vastly
        // out-appear the tail.
        let selection_weights: Vec<f64> = (0..12).map(|i| if i < 3 { 1.0 } else { 0.0001 }).collect();
        let ratings: Vec<f64> = (0..12).map(|i| if i < 3 { 5.0 } else { 0.0 }).collect();
        let mut rng = make_rng(Some(4), crate::seed::SUBSYSTEM_PAIRING);
        let lineups = generate_top_heavy_lineups_indexed(12, 200, 3, &selection_weights, &ratings, None, 1.0, &mut rng);
        let mut appearances = [0usize; 12];
        for item in lineups.iter().flatten() {
            appearances[*item] += 1;
        }
        let top_3: usize = appearances[0] + appearances[1] + appearances[2];
        let tail: usize = appearances[3..].iter().sum();
        assert!(top_3 > tail, "contenders ({top_3}) should out-appear tail ({tail})");
    }

    #[test]
    fn test_lineups_empty_below_lineup_size() {
        for lineup_size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let num_items = lineup_size - 1;
            let ratings = vec![1.0; num_items];
            let edges = vec![0usize; num_items];
            let weights = vec![1.0; num_items];
            let mut rng = make_rng(Some(5), crate::seed::SUBSYSTEM_PAIRING);
            assert!(
                generate_uniform_lineups_indexed(
                    num_items, 4, lineup_size, &ratings, None, 1.0, &edges, &mut rng,
                ).is_empty(),
                "size {lineup_size}: uniform lineups should be empty with {num_items} items"
            );
            assert!(
                generate_top_heavy_lineups_indexed(
                    num_items, 4, lineup_size, &weights, &ratings, None, 1.0, &mut rng,
                ).is_empty(),
                "size {lineup_size}: top-heavy lineups should be empty with {num_items} items"
            );
        }
    }

    #[test]
    fn test_top_heavy_single_weighted_item_is_not_degenerate() {
        // A single item holding all the selection weight used to be a hard error
        // (item2 was drawn from the remaining weight). Now item2 comes from the
        // rating window instead, so item1 is always item 0 and it still gets a
        // valid, distinct opponent.
        let item_ids: Vec<i64> = (0..3).collect();
        let selection_weights = vec![1.0, 0.0, 0.0];
        let ratings = vec![0.0, 1.0, 2.0];
        let zeros = vec![0usize; 3];

        let pairs = generate_top_heavy_pairings(&item_ids, 5, &selection_weights, &ratings, None, 1.0, &zeros, &zeros);
        assert_eq!(pairs.len(), 5);
        for (a, b) in &pairs {
            assert_ne!(a, b);
            assert!(*a == 0 || *b == 0, "item 0 holds all selection weight; it must be in every pair");
        }
    }
}
