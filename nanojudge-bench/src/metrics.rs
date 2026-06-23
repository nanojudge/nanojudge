use std::collections::HashMap;

/// Spearman rank correlation coefficient between two orderings of the same items.
///
/// Returns a value in [-1, 1] where 1 = identical ranking, -1 = perfectly reversed.
pub fn spearman_rho(true_order: &[String], output_order: &[String]) -> f64 {
    let n = true_order.len();
    if n <= 1 {
        return 1.0;
    }

    let output_ranks: HashMap<&str, usize> = output_order
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i + 1))
        .collect();

    let d_squared_sum: f64 = true_order
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let true_rank = (i + 1) as f64;
            let output_rank = output_ranks[name.as_str()] as f64;
            (true_rank - output_rank).powi(2)
        })
        .sum();

    let n = n as f64;
    1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1.0))
}

/// Mean absolute rank displacement for the true top-K items.
///
/// For each of the K highest-strength items, computes how many positions away
/// from its true rank it landed in the output, then averages. 0.0 = perfect.
pub fn top_k_displacement(true_order: &[String], output_order: &[String], k: usize) -> f64 {
    let k = k.min(true_order.len()).min(output_order.len());
    if k == 0 {
        return 0.0;
    }

    let output_ranks: HashMap<&str, usize> = output_order
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let total: f64 = true_order[..k]
        .iter()
        .enumerate()
        .map(|(true_rank, name)| {
            let output_rank = output_ranks[name.as_str()];
            (true_rank as f64 - output_rank as f64).abs()
        })
        .sum();

    total / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(ids: &[usize]) -> Vec<String> {
        ids.iter().map(|i| format!("item_{i:04}")).collect()
    }

    #[test]
    fn identical_rankings() {
        let order = names(&[0, 1, 2, 3, 4]);
        assert!((spearman_rho(&order, &order) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reversed_rankings() {
        let true_order = names(&[0, 1, 2, 3, 4]);
        let output_order = names(&[4, 3, 2, 1, 0]);
        assert!((spearman_rho(&true_order, &output_order) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn single_item() {
        let order = names(&[0]);
        assert!((spearman_rho(&order, &order) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn displacement_perfect() {
        let order = names(&[0, 1, 2, 3, 4]);
        assert!((top_k_displacement(&order, &order, 3) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn displacement_top1_off_by_two() {
        let true_order = names(&[0, 1, 2, 3, 4]);
        let output_order = names(&[1, 2, 0, 3, 4]);
        // True #1 (item_0000) is at output position 2 → displacement 2.
        assert!((top_k_displacement(&true_order, &output_order, 1) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn displacement_top3_average() {
        let true_order = names(&[0, 1, 2, 3, 4]);
        let output_order = names(&[1, 0, 3, 2, 4]);
        // item_0000: true 0, output 1 → 1
        // item_0001: true 1, output 0 → 1
        // item_0002: true 2, output 3 → 1
        // Average: 3/3 = 1.0
        assert!((top_k_displacement(&true_order, &output_order, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn displacement_clamped_to_length() {
        let order = names(&[0, 1, 2]);
        assert!((top_k_displacement(&order, &order, 100) - 0.0).abs() < 1e-12);
    }
}
