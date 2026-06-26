/// Comparison distributions for pairwise comparison tournaments.
///
/// Public functions accept `item_ids: &[i64]` and return `Pair` (i64, i64).
/// Internal functions use `usize` indices for efficient array indexing.
use rand::Rng;
use rand::rngs::StdRng;

use crate::constants::OPPONENT_WINDOW_SIZE;
use crate::seed::make_rng;
use crate::types::{IndexedPair, Pair};

/// Comparison distribution enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComparisonDistribution {
    Uniform,
    TopHeavy,
}

/// Calculate information gain for a matchup between two items.
pub fn calculate_info_gain(rating_a: f64, rating_b: f64, sharpness: f64) -> f64 {
    let p = 1.0 / (1.0 + (rating_b - rating_a).exp());
    let info_gain = p * (1.0 - p);
    info_gain.powf(sharpness)
}

/// Probability that item A goes in position 1 against item B.
///
/// Uses Laplace-smoothed first-position ratios so that an item which has
/// historically gone first more often gets a lower probability of going
/// first next time. With no history (zero counts) this returns 0.5,
/// degrading gracefully to a fair coin flip. The +1 / +2 smoothing damps
/// extreme ratios from small samples.
fn position_probability(
    first_count_a: usize, games_a: usize,
    first_count_b: usize, games_b: usize,
) -> f64 {
    let ratio_a = (first_count_a as f64 + 1.0) / (games_a as f64 + 2.0);
    let ratio_b = (first_count_b as f64 + 1.0) / (games_b as f64 + 2.0);
    ratio_b / (ratio_a + ratio_b)
}

/// Determine the effective comparison distribution to use for this round.
///
/// Two stages:
///   Stage 1 (Warm-up): Uniform until every item has >= min_uniform_games.
///   Stage 2 (Main phase): Top-heavy.
///
/// # Panics
///
/// Panics if `games_played` has fewer than `num_items` entries.
pub fn get_effective_comparison_distribution(
    user_comparison_distribution: ComparisonDistribution,
    num_items: usize,
    games_played: &[usize],
    min_uniform_games: usize,
) -> ComparisonDistribution {
    if user_comparison_distribution == ComparisonDistribution::Uniform {
        return ComparisonDistribution::Uniform;
    }

    // Stage 1: Warm-up
    for &games in games_played.iter().take(num_items) {
        if games < min_uniform_games {
            return ComparisonDistribution::Uniform;
        }
    }

    // Stage 2: Main phase
    ComparisonDistribution::TopHeavy
}

// ---------------------------------------------------------------------------
// Public pairing functions (work with i64 IDs)
// ---------------------------------------------------------------------------

/// Generate uniform pairings for a round.
///
/// `current_ratings[i]` is the rating for `item_ids[i]`.
/// Returns pairs of item IDs.
///
/// # Panics
///
/// Panics if `current_ratings` has fewer entries than `item_ids`.
pub fn generate_uniform_pairings(
    item_ids: &[i64],
    pairs_count: usize,
    current_ratings: &[f64],
    sharpness: f64,
) -> Vec<Pair> {
    let n = item_ids.len();
    let zeros = vec![0usize; n];
    let mut rng = make_rng(None, crate::seed::SUBSYSTEM_PAIRING);
    let index_pairs = generate_uniform_pairings_indexed(
        n,
        pairs_count,
        current_ratings,
        sharpness,
        &zeros,
        &zeros,
        &mut rng,
    );
    index_pairs.into_iter().map(|(a, b)| (item_ids[a], item_ids[b])).collect()
}

/// Generate top-heavy pairings for a round.
///
/// Both items in every pair are sampled from the per-item selection weights
/// `selection_weights[i]` (corresponding to `item_ids[i]`), so contenders are
/// compared against contenders. Returns pairs of item IDs.
///
/// # Panics
///
/// Panics if `selection_weights` length does not equal `item_ids` length, or if
/// after picking the first item no weight remains for a distinct second item
/// (only possible with a degenerate list where a single item holds all the
/// weight).
pub fn generate_top_heavy_pairings(
    item_ids: &[i64],
    pairs_count: usize,
    selection_weights: &[f64],
) -> Vec<Pair> {
    let n = item_ids.len();
    let zeros = vec![0usize; n];
    let mut rng = make_rng(None, crate::seed::SUBSYSTEM_PAIRING);
    let index_pairs = generate_top_heavy_pairings_indexed(
        n,
        pairs_count,
        selection_weights,
        &zeros,
        &zeros,
        &mut rng,
    );
    index_pairs.into_iter().map(|(a, b)| (item_ids[a], item_ids[b])).collect()
}

// ---------------------------------------------------------------------------
// Internal indexed pairing functions (work with usize indices)
// ---------------------------------------------------------------------------

pub(crate) fn generate_uniform_pairings_indexed(
    num_items: usize,
    pairs_count: usize,
    current_ratings: &[f64],
    sharpness: f64,
    first_position_counts: &[usize],
    games_played: &[usize],
    rng: &mut StdRng,
) -> Vec<IndexedPair> {
    let mut pairings: Vec<IndexedPair> = Vec::with_capacity(pairs_count);

    if num_items < 2 {
        return pairings;
    }

    // Local mutable copies of caller-provided counters. Updated optimistically
    // as positions are assigned within this call so later pairs balance against
    // earlier ones. Discarded at function return — caller state is never mutated.
    let mut local_first_counts: Vec<usize> = first_position_counts.to_vec();
    let mut local_games: Vec<usize> = games_played.to_vec();

    generate_uniform_iteration(
        num_items, current_ratings, sharpness, pairs_count,
        &mut pairings, rng,
        &mut local_first_counts, &mut local_games,
    );

    pairings
}

#[allow(clippy::too_many_arguments)]
fn generate_uniform_iteration(
    num_items: usize,
    current_ratings: &[f64],
    sharpness: f64,
    max_pairs: usize,
    pairings: &mut Vec<IndexedPair>,
    rng: &mut impl Rng,
    first_counts: &mut [usize],
    total_games: &mut [usize],
) {
    // Sort items by rating ascending so we can pick opponents from a narrow
    // rating-window around each item1. sorted_pool is immutable for the rest
    // of the function; "removal" is done via tombstones so the remaining
    // entries keep their sorted positions (and the window math stays valid).
    let mut sorted_pool: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, current_ratings[i]))
        .collect();
    sorted_pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_pool.len();
    // alive[p] = false means sorted_pool[p] has already been paired this iteration.
    let mut alive = vec![true; n];
    // live_positions holds the sorted-pool indices still available. Picking
    // item1 is a swap_remove on this list (O(1) random removal). live_idx_of
    // is the inverse map: live_idx_of[p] = index of p inside live_positions,
    // or None if tombstoned. It lets us also O(1)-remove item2 once we know
    // its sorted-pool position.
    let mut live_positions: Vec<usize> = (0..n).collect();
    let mut live_idx_of: Vec<Option<usize>> = (0..n).map(Some).collect();

    let half_w = OPPONENT_WINDOW_SIZE / 2;
    let mut weights: Vec<f64> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 1);
    let mut candidates: Vec<usize> = Vec::with_capacity(OPPONENT_WINDOW_SIZE + 1);

    for _ in 0..max_pairs {
        if live_positions.len() < 2 {
            break;
        }

        // Pick item1: random live entry, removed via swap_remove.
        let live_idx1 = rng.random_range(0..live_positions.len());
        let pos1 = swap_remove_live(&mut live_positions, &mut live_idx_of, live_idx1);
        alive[pos1] = false;
        let (item1, item1_rating) = sorted_pool[pos1];

        // Collect live candidates in the rating window around pos1.
        let window_start = pos1.saturating_sub(half_w);
        let window_end = (pos1 + half_w + 1).min(n);
        candidates.clear();
        weights.clear();
        for p in window_start..window_end {
            if alive[p] {
                candidates.push(p);
                weights.push(calculate_info_gain(item1_rating, sorted_pool[p].1, sharpness));
            }
        }
        if candidates.is_empty() {
            // Window exhausted — skip this item1 for the rest of this iteration.
            continue;
        }

        let total_weight: f64 = weights.iter().sum();
        let selected = if total_weight == 0.0 {
            rng.random_range(0..candidates.len())
        } else {
            weighted_random_select(&weights, total_weight, rng)
        };

        let pos2 = candidates[selected];
        let live_idx2 = live_idx_of[pos2].expect("candidate must be alive");
        swap_remove_live(&mut live_positions, &mut live_idx_of, live_idx2);
        alive[pos2] = false;
        let (item2, _) = sorted_pool[pos2];

        let p = position_probability(
            first_counts[item1], total_games[item1],
            first_counts[item2], total_games[item2],
        );
        if rng.random::<f64>() < p {
            pairings.push((item1, item2));
            first_counts[item1] += 1;
        } else {
            pairings.push((item2, item1));
            first_counts[item2] += 1;
        }
        total_games[item1] += 1;
        total_games[item2] += 1;
    }
}

/// Remove the entry at `live_idx` from `live_positions` in O(1) using
/// swap_remove, keeping `live_idx_of` in sync. Returns the sorted-pool
/// position that was removed.
fn swap_remove_live(
    live_positions: &mut Vec<usize>,
    live_idx_of: &mut [Option<usize>],
    live_idx: usize,
) -> usize {
    let removed_pos = live_positions.swap_remove(live_idx);
    live_idx_of[removed_pos] = None;
    // If we didn't remove the last entry, the old last entry now sits at live_idx.
    if live_idx < live_positions.len() {
        let moved_pos = live_positions[live_idx];
        live_idx_of[moved_pos] = Some(live_idx);
    }
    removed_pos
}

pub(crate) fn generate_top_heavy_pairings_indexed(
    num_items: usize,
    pairs_count: usize,
    selection_weights: &[f64],
    first_position_counts: &[usize],
    games_played: &[usize],
    rng: &mut StdRng,
) -> Vec<IndexedPair> {
    if num_items < 2 {
        return Vec::new();
    }
    assert_eq!(selection_weights.len(), num_items, "selection_weights length mismatch");
    let pairs_target = pairs_count;

    let total_weight: f64 = selection_weights.iter().sum();
    assert!(
        total_weight > 0.0,
        "top-heavy pairing requires positive total item-selection weight"
    );

    // Local mutable copies of caller-provided counters. Updated optimistically
    // as positions are assigned within this call so later pairs balance against
    // earlier ones. Discarded at function return — caller state is never mutated.
    let mut local_first_counts: Vec<usize> = first_position_counts.to_vec();
    let mut local_games: Vec<usize> = games_played.to_vec();

    let mut pairs: Vec<IndexedPair> = Vec::with_capacity(pairs_target);

    for _ in 0..pairs_target {
        // Both items are sampled from the layered weights: item1 from the full
        // distribution, item2 from the same distribution with item1 removed.
        let item1 = weighted_random_select(selection_weights, total_weight, rng);

        // Remaining weight after excluding item1. Zero here means a single item
        // holds all the weight — a degenerate list — so we panic rather than
        // fabricate an opponent.
        let total_remaining = total_weight - selection_weights[item1];
        assert!(
            total_remaining > 0.0,
            "top-heavy pairing: all item-selection weight is on one item; cannot pick a distinct opponent"
        );
        let item2 = weighted_random_select_excluding(selection_weights, total_remaining, item1, rng);

        let p = position_probability(
            local_first_counts[item1], local_games[item1],
            local_first_counts[item2], local_games[item2],
        );
        if rng.random::<f64>() < p {
            pairs.push((item1, item2));
            local_first_counts[item1] += 1;
        } else {
            pairs.push((item2, item1));
            local_first_counts[item2] += 1;
        }
        local_games[item1] += 1;
        local_games[item2] += 1;
    }

    pairs
}

fn weighted_random_select(weights: &[f64], total_weight: f64, rng: &mut impl Rng) -> usize {
    let mut r = rng.random::<f64>() * total_weight;
    for (j, &w) in weights.iter().enumerate() {
        r -= w;
        if r < 1e-10 {
            return j;
        }
    }
    weights.len() - 1
}

/// Weighted selection over `weights`, skipping index `exclude`. `total_weight`
/// must be the sum of all weights except `weights[exclude]` (i.e. the caller has
/// already subtracted the excluded item's weight) and must be positive.
fn weighted_random_select_excluding(
    weights: &[f64],
    total_weight: f64,
    exclude: usize,
    rng: &mut impl Rng,
) -> usize {
    let mut r = rng.random::<f64>() * total_weight;
    let mut last = exclude;
    for (j, &w) in weights.iter().enumerate() {
        if j == exclude {
            continue;
        }
        last = j;
        r -= w;
        if r < 1e-10 {
            return j;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_effective_comparison_distribution_uniform_user_choice() {
        let games = vec![10, 10];
        let result = get_effective_comparison_distribution(ComparisonDistribution::Uniform, 2, &games, 3);
        assert_eq!(result, ComparisonDistribution::Uniform);
    }

    #[test]
    fn test_effective_comparison_distribution_warmup_stage() {
        let games = vec![1, 10]; // Item 0 below minimum
        let result = get_effective_comparison_distribution(ComparisonDistribution::TopHeavy, 2, &games, 3);
        assert_eq!(result, ComparisonDistribution::Uniform);
    }

    #[test]
    fn test_effective_comparison_distribution_main_phase() {
        let games = vec![10, 10];
        let result = get_effective_comparison_distribution(ComparisonDistribution::TopHeavy, 2, &games, 3);
        assert_eq!(result, ComparisonDistribution::TopHeavy);
    }

    #[test]
    fn test_uniform_pairings_coverage() {
        let item_ids: Vec<i64> = (100..110).collect(); // IDs 100-109
        let ratings = vec![1.0; 10];
        let pairs = generate_uniform_pairings(&item_ids, 5, &ratings, 1.0);

        assert_eq!(pairs.len(), 5);

        // All pairs should use IDs from item_ids, not indices
        for (a, b) in &pairs {
            assert!(*a >= 100 && *a <= 109, "ID {} not in range", a);
            assert!(*b >= 100 && *b <= 109, "ID {} not in range", b);
        }
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
        // A has played 1 game and went first. Without smoothing this would be
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
    fn test_top_heavy_pairings() {
        let item_ids: Vec<i64> = (0..10).collect();
        let selection_weights: Vec<f64> = (0..10).map(|i| if i < 3 { 0.8 } else { 0.05 }).collect();

        let pairs = generate_top_heavy_pairings(&item_ids, 5, &selection_weights);
        assert_eq!(pairs.len(), 5);
        for (a, b) in &pairs {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn test_top_heavy_both_items_from_weights() {
        // Items 0,1,2 carry essentially all the weight; 3-9 carry ~none. Both
        // slots should be drawn from the heavy items, so the contenders vastly
        // out-appear the tail across a long batch.
        let item_ids: Vec<i64> = (0..10).collect();
        let selection_weights: Vec<f64> = (0..10).map(|i| if i < 3 { 1.0 } else { 0.0001 }).collect();

        let pairs = generate_top_heavy_pairings(&item_ids, 200, &selection_weights);
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
    #[should_panic(expected = "all item-selection weight is on one item")]
    fn test_top_heavy_panics_when_one_item_holds_all_weight() {
        let item_ids: Vec<i64> = (0..3).collect();
        let selection_weights = vec![1.0, 0.0, 0.0];
        let _ = generate_top_heavy_pairings(&item_ids, 1, &selection_weights);
    }
}
