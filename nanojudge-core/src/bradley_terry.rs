/// Iterative Maximum Likelihood Estimation for Bradley-Terry model.
///
/// Uses ghost player regularization and fractional wins from logprob-derived probabilities.
/// Internal module — operates on pre-mapped `usize` indices, not caller IDs.
use std::collections::HashMap;

use crate::types::IndexedComparison;

const CONVERGENCE_THRESHOLD: f64 = 1e-6;

pub struct BradleyTerry {
    /// Number of real items (excluding ghost).
    num_items: usize,
    /// Total number of items including ghost player.
    total: usize,
    /// Sparse fractional wins: wins_table[i] maps opponent index -> fractional wins of i over opponent.
    wins_table: Vec<HashMap<usize, f64>>,
    /// Total wins per item (precomputed for efficiency).
    total_wins: Vec<f64>,
    /// Current scores (indices 0..num_items are real items, last is ghost).
    pub scores: Vec<f64>,
}

impl BradleyTerry {
    pub fn new(
        num_items: usize,
        results: &[IndexedComparison],
        regularization_strength: f64,
    ) -> Self {
        let ghost_idx = num_items;
        let total = num_items + 1; // real items + ghost

        // Build sparse wins table
        let mut wins_table: Vec<HashMap<usize, f64>> = (0..total).map(|_| HashMap::new()).collect();

        for &(i1, i2, prob) in results {
            assert!(i1 < num_items, "item1 index {} out of range (num_items = {})", i1, num_items);
            assert!(i2 < num_items, "item2 index {} out of range (num_items = {})", i2, num_items);

            *wins_table[i1].entry(i2).or_insert(0.0) += prob;
            *wins_table[i2].entry(i1).or_insert(0.0) += 1.0 - prob;
        }

        // Ghost player regularization (O(n) instead of O(n²))
        if regularization_strength > 0.0 {
            for i in 0..num_items {
                *wins_table[i].entry(ghost_idx).or_insert(0.0) += regularization_strength;
                *wins_table[ghost_idx].entry(i).or_insert(0.0) += regularization_strength;
            }
        }

        // Precompute total wins per item
        let mut total_wins = vec![0.0; total];
        for i in 0..total {
            let mut sum = 0.0;
            for (&j, &w) in &wins_table[i] {
                if j != i {
                    sum += w;
                }
            }
            total_wins[i] = sum;
        }

        BradleyTerry {
            num_items,
            total,
            wins_table,
            total_wins,
            scores: vec![1.0; total],
        }
    }

    fn get_wins(&self, i: usize, j: usize) -> f64 {
        self.wins_table[i].get(&j).copied().unwrap_or(0.0)
    }

    fn run_iteration(&mut self) {
        let mut new_scores = vec![0.0; self.total];

        for i in 0..self.total {
            let total_wins_i = self.total_wins[i];

            if total_wins_i == 0.0 {
                new_scores[i] = 0.0;
                continue;
            }

            let score_i = self.scores[i];
            let mut denominator = 0.0;

            // Only iterate over items that i has actually played against (sparse)
            for (&j, &wins_i_to_j) in &self.wins_table[i] {
                if j == i {
                    continue;
                }

                let score_j = self.scores[j];
                let wins_j_to_i = self.get_wins(j, i);
                let total_games = wins_i_to_j + wins_j_to_i;

                if total_games > 0.0 && (score_i + score_j) > 0.0 {
                    denominator += total_games / (score_i + score_j);
                }
            }

            new_scores[i] = if denominator > 0.0 {
                total_wins_i / denominator
            } else {
                self.scores[i]
            };
        }

        self.scores = new_scores;
    }

    /// Normalize scores by dividing by the geometric mean.
    /// Items with 0 wins stay at 0 (standard BT behavior).
    fn normalize_scores(&mut self) {
        let mut log_sum = 0.0;
        let mut non_zero_count = 0usize;

        for &score in &self.scores {
            if score > 0.0 {
                log_sum += score.ln();
                non_zero_count += 1;
            }
        }

        if non_zero_count == 0 {
            return;
        }

        let log_geo_mean = log_sum / non_zero_count as f64;
        let geo_mean = log_geo_mean.exp();

        if geo_mean > 0.0 {
            for score in &mut self.scores {
                *score /= geo_mean;
            }
        }
    }

    /// Run iterative BT score calculation with early convergence stopping.
    pub fn calculate_scores(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let old_scores = self.scores.clone();
            self.run_iteration();
            self.normalize_scores();

            let max_change = self
                .scores
                .iter()
                .zip(old_scores.iter())
                .map(|(new, old)| (new - old).abs())
                .fold(0.0_f64, f64::max);

            if max_change < CONVERGENCE_THRESHOLD {
                break;
            }
        }
    }

    /// Get score for a specific item by index.
    pub fn get_score(&self, item: usize) -> f64 {
        self.scores[item]
    }

    /// Get scores for all real items (excluding ghost), as a slice.
    pub fn real_scores(&self) -> &[f64] {
        &self.scores[..self.num_items]
    }

    /// Number of real items (excluding ghost).
    pub fn num_items(&self) -> usize {
        self.num_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_comp(i1: usize, i2: usize, prob: f64) -> IndexedComparison {
        (i1, i2, prob)
    }

    #[test]
    fn test_basic_ranking() {
        // 3 items: 0=A, 1=B, 2=C
        let results = vec![
            make_comp(0, 1, 0.9), // A strongly beats B
            make_comp(0, 2, 0.8), // A beats C
            make_comp(1, 2, 0.7), // B beats C
        ];

        let mut bt = BradleyTerry::new(3, &results, 0.01);
        bt.calculate_scores(30);

        let scores = bt.real_scores();
        // A > B > C
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_no_comparisons_equal_scores() {
        let results: Vec<IndexedComparison> = vec![];

        let mut bt = BradleyTerry::new(2, &results, 0.01);
        bt.calculate_scores(30);

        let scores = bt.real_scores();
        let diff = (scores[0] - scores[1]).abs();
        assert!(diff < 0.01, "Scores should be nearly equal with no comparisons");
    }

    #[test]
    fn test_geometric_mean_normalization() {
        let results = vec![make_comp(0, 1, 0.7)];

        let mut bt = BradleyTerry::new(2, &results, 0.01);
        bt.calculate_scores(30);

        let scores = bt.real_scores();
        let product: f64 = scores.iter().product();
        let geo_mean = product.powf(1.0 / scores.len() as f64);
        assert!(
            (geo_mean - 1.0).abs() < 0.1,
            "Geometric mean should be close to 1.0, got {}",
            geo_mean
        );
    }
}
