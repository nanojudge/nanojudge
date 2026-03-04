/// Initial Bradley-Terry rating assigned to all items at the start.
/// Using 1.0 as the neutral starting point allows the algorithm to converge naturally.
pub const INITIAL_BRADLEY_TERRY_RATING: f64 = 1.0;

/// Maximum number of concurrent comparison requests to process at once.
/// Used by `calculate_pairs_for_round` to determine progressive scaling.
pub const CONCURRENCY_LIMIT: usize = 300;

/// Maximum multiplier for scaling pairs per round in small lists.
/// For lists where base_pairs < CONCURRENCY_LIMIT, we progressively scale
/// the number of pairs per round (round 1: 1x, round 2: 2x, etc.) up to this max.
pub const MAX_ROUND_MULTIPLIER: usize = 4;

/// Maximum number of nearby items to consider when selecting an opponent.
///
/// When choosing who to pair an item against, we use info-gain weighting:
/// items with similar ratings produce more informative comparisons.
/// Info-gain drops off sharply with rating distance, so scanning all N items
/// is wasteful — the far-away ones almost never get selected.
///
/// Instead, we sort items by rating, then only consider the closest
/// OPPONENT_WINDOW_SIZE items. This turns opponent selection from O(N) to O(1)
/// per pair, making overall pairing O(N log N) instead of O(N^2).
///
/// For lists smaller than this value, all items are considered (no change
/// in behavior). The value 100 provides more than enough candidates for
/// good-quality weighted sampling while keeping pairing fast at any scale.
///
/// Benchmarked on 2025-02-25 (see misc/benchmarks/mcmc-bench):
///   10K items: 452ms (old O(N^2)) vs ~5ms with windowing
///   100K items: 46s (old O(N^2)) vs ~50ms with windowing
pub const OPPONENT_WINDOW_SIZE: usize = 100;
