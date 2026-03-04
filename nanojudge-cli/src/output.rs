/// Output formatting: terminal table and JSON.
use nanojudge_core::RankedItem;
use serde::Serialize;

#[derive(Serialize)]
struct JsonRankedItem {
    rank: usize,
    name: String,
    score: f64,
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Serialize)]
struct JsonOutput {
    items: Vec<JsonRankedItem>,
    total_comparisons: usize,
    rounds: usize,
}

/// Print results as a formatted terminal table.
pub fn print_table(rankings: &[RankedItem], names: &[String], games_played: &[usize], rounds: usize, total_comparisons: usize, positional_bias: f64, positional_bias_confidence_interval: (f64, f64)) {
    // Find the widest item name for padding
    let name_width = rankings.iter()
        .map(|r| names[r.item as usize].len())
        .max()
        .unwrap_or(4)
        .max(4); // at least "Item"

    // Header
    println!(" # | {:<name_width$} |   Score | 95% CI Low | 95% CI High | Comparisons", "Item");
    println!("---|-{}-|---------|------------|-------------|------------", "-".repeat(name_width));

    // Rows
    for (i, r) in rankings.iter().enumerate() {
        let name = &names[r.item as usize];
        let games = games_played[r.item as usize];
        println!(
            "{:>2} | {:<name_width$} | {:>7.4} | {:>10.2} | {:>11.2} | {:>11}",
            i + 1, name, r.score, r.lower_bound, r.upper_bound, games,
        );
    }

    println!(
        "\n{} items ranked across {} rounds ({} comparisons)",
        rankings.len(),
        rounds,
        total_comparisons,
    );
    println!(
        "Position bias — estimated: {:.3} [{:.3}, {:.3}] (corrected for in scores, 0.5 = no bias)",
        positional_bias, positional_bias_confidence_interval.0, positional_bias_confidence_interval.1,
    );
}

/// Print results as JSON.
pub fn print_json(rankings: &[RankedItem], names: &[String], rounds: usize, total_comparisons: usize) {
    let items: Vec<JsonRankedItem> = rankings
        .iter()
        .enumerate()
        .map(|(i, r)| JsonRankedItem {
            rank: i + 1,
            name: names[r.item as usize].clone(),
            score: r.score,
            lower_bound: r.lower_bound,
            upper_bound: r.upper_bound,
        })
        .collect();

    let output = JsonOutput {
        items,
        total_comparisons,
        rounds,
    };

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
