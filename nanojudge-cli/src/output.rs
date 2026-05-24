/// Output formatting: terminal table and JSON.
use nanojudge_core::{JudgeAnalytics, RankedItem};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
struct JsonRankedItem {
    rank: usize,
    id: i64,
    name: String,
    score: f64,
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Serialize)]
struct JsonJudgeAnalytics {
    judge_id: u64,
    positional_bias: f64,
    positional_bias_ci_low: f64,
    positional_bias_ci_high: f64,
    decisiveness: Option<f64>,
    decisiveness_ci_low: Option<f64>,
    decisiveness_ci_high: Option<f64>,
    num_comparisons: usize,
}

#[derive(Serialize)]
struct JsonOutput {
    items: Vec<JsonRankedItem>,
    total_comparisons: usize,
    rounds: usize,
    positional_bias: f64,
    positional_bias_ci_low: f64,
    positional_bias_ci_high: f64,
    judge_analytics: Vec<JsonJudgeAnalytics>,
}

/// Print results as a formatted terminal table.
pub fn print_table(
    rankings: &[RankedItem],
    names: &[String],
    games_played: &[usize],
    rounds: usize,
    total_comparisons: usize,
    confidence_level: f64,
    judge_analytics: &[JudgeAnalytics],
    judge_names: &HashMap<u64, String>,
    judge_tokens: &HashMap<u64, (u64, u64)>,
    judge_avg_wall_time: &HashMap<u64, f64>,
) {
    let ci_label = format!("{:.0}% CI", confidence_level * 100.0);

    // Find the widest item name for padding
    let name_width = rankings
        .iter()
        .map(|r| names[r.item as usize].len())
        .max()
        .unwrap_or(4)
        .max(4); // at least "Item"

    // Header
    println!(
        " # | {:<name_width$} |   Score | {:>10} | {:>11} | Comparisons | ID",
        "Item",
        format!("{} Low", ci_label),
        format!("{} High", ci_label),
    );
    println!(
        "---|-{}-|---------|------------|-------------|-------------|----",
        "-".repeat(name_width)
    );

    // Rows
    for (i, r) in rankings.iter().enumerate() {
        let name = &names[r.item as usize];
        let games = games_played[r.item as usize];
        println!(
            "{:>2} | {:<name_width$} | {:>7.4} | {:>10.2} | {:>11.2} | {:>11} | {:>2}",
            i + 1,
            name,
            r.score,
            r.lower_bound,
            r.upper_bound,
            games,
            r.item,
        );
    }

    println!(
        "\n{} items ranked across {} rounds ({} comparisons)",
        rankings.len(),
        rounds,
        total_comparisons,
    );

    // Print per-judge analytics
    if judge_analytics.len() == 1 {
        let ja = &judge_analytics[0];
        println!(
            "Position bias — estimated: {:.3} [{:.3}, {:.3}] (corrected for in scores, 0.5 = no bias)",
            ja.positional_bias, ja.positional_bias_ci.0, ja.positional_bias_ci.1,
        );
        if let Some(&(input, output)) = judge_tokens.get(&ja.judge_id) {
            if input > 0 || output > 0 {
                println!(
                    "Tokens — input: {}, output: {}",
                    format_count(input as usize),
                    format_count(output as usize)
                );
            }
        }
        if let Some(&avg) = judge_avg_wall_time.get(&ja.judge_id) {
            if avg > 0.0 {
                println!("Avg wall time per round: {}", format_duration(avg));
            }
        }
    } else {
        print_judge_panel_analytics(
            judge_analytics,
            judge_names,
            judge_tokens,
            judge_avg_wall_time,
        );
    }
}

/// Print the judge panel analytics table (design doc section 9 format).
fn print_judge_panel_analytics(
    analytics: &[JudgeAnalytics],
    judge_names: &HashMap<u64, String>,
    judge_tokens: &HashMap<u64, (u64, u64)>,
    judge_avg_wall_time: &HashMap<u64, f64>,
) {
    let has_decisiveness = analytics.iter().any(|ja| ja.decisiveness.is_some());
    let has_tokens = analytics.iter().any(|ja| {
        judge_tokens
            .get(&ja.judge_id)
            .is_some_and(|&(i, o)| i > 0 || o > 0)
    });
    let has_wall_time = analytics.iter().any(|ja| {
        judge_avg_wall_time
            .get(&ja.judge_id)
            .is_some_and(|&t| t > 0.0)
    });

    // Find the widest judge name for padding
    let name_width = analytics
        .iter()
        .map(|ja| judge_names.get(&ja.judge_id).map_or(16, |n| n.len()))
        .max()
        .unwrap_or(5)
        .max(5); // at least "Judge"

    println!();

    // Header
    let mut header = format!(
        "  {:<name_width$}   {:>11}   {:>15}",
        "Judge", "Comparisons", "Bias (->item1)"
    );
    let mut separator = format!(
        "  {:<name_width$}   {:>11}   {:>15}",
        "\u{2500}".repeat(name_width.min(30)),
        "\u{2500}".repeat(11),
        "\u{2500}".repeat(15)
    );

    if has_decisiveness {
        header += &format!("   {:>18}", "Decisiveness");
        separator += &format!("   {:>18}", "\u{2500}".repeat(18));
    }
    if has_tokens {
        header += &format!("   {:>13}   {:>13}", "Input tokens", "Output tokens");
        separator += &format!(
            "   {:>13}   {:>13}",
            "\u{2500}".repeat(13),
            "\u{2500}".repeat(13)
        );
    }
    if has_wall_time {
        header += &format!("   {:>14}", "Avg round time");
        separator += &format!("   {:>14}", "\u{2500}".repeat(14));
    }

    println!("{header}");
    println!("{separator}");

    for ja in analytics {
        let name = judge_names
            .get(&ja.judge_id)
            .cloned()
            .unwrap_or_else(|| format!("{:016x}", ja.judge_id));
        let bias_str = format!(
            "{:.2} [{:.2}-{:.2}]",
            ja.positional_bias, ja.positional_bias_ci.0, ja.positional_bias_ci.1,
        );

        let mut line = format!(
            "  {:<name_width$}   {:>11}   {:>15}",
            name,
            format_count(ja.num_comparisons),
            bias_str,
        );

        if has_decisiveness {
            let dec_str = if let Some(d) = ja.decisiveness {
                let (lo, hi) = ja.decisiveness_ci.unwrap_or((d, d));
                format!("{:.2} [{:.2}-{:.2}]", d, lo, hi)
            } else {
                "n/a".to_string()
            };
            line += &format!("   {:>18}", dec_str);
        }

        if has_tokens {
            let (input, output) = judge_tokens.get(&ja.judge_id).copied().unwrap_or((0, 0));
            line += &format!(
                "   {:>13}   {:>13}",
                format_count(input as usize),
                format_count(output as usize)
            );
        }

        if has_wall_time {
            let avg = judge_avg_wall_time
                .get(&ja.judge_id)
                .copied()
                .unwrap_or(0.0);
            line += &format!("   {:>14}", format_duration(avg));
        }

        println!("{line}");
    }

    if has_decisiveness {
        println!("\n  Panel average decisiveness: 1.00 (by definition)");
    }
}

/// Print a compact live ranking table to stderr after a round.
/// If `limit` is 0, prints all items.
pub fn print_live_table(
    rankings: &[RankedItem],
    names: &[String],
    round: usize,
    total_comparisons: usize,
    limit: usize,
) {
    let show = if limit == 0 || limit >= rankings.len() {
        rankings
    } else {
        &rankings[..limit]
    };

    let name_width = show
        .iter()
        .map(|r| names[r.item as usize].len())
        .max()
        .unwrap_or(4)
        .max(4);

    eprintln!("\n── Round {} ({} comparisons) ──", round, total_comparisons);
    for (i, r) in show.iter().enumerate() {
        eprintln!(
            " {:>2}. {:<name_width$}  {:>7.4}  [{:.2}, {:.2}]",
            i + 1,
            names[r.item as usize],
            r.score,
            r.lower_bound,
            r.upper_bound,
        );
    }
    if limit > 0 && limit < rankings.len() {
        eprintln!("     … and {} more", rankings.len() - limit);
    }
}

/// Format a duration in seconds to a human-readable string.
fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else {
        let mins = (secs / 60.0).floor() as u64;
        let remaining = secs - (mins as f64 * 60.0);
        format!("{}m {:.1}s", mins, remaining)
    }
}

/// Format a number with comma separators for readability.
fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Build JSON output string.
fn build_json(
    rankings: &[RankedItem],
    names: &[String],
    rounds: usize,
    total_comparisons: usize,
    judge_analytics: &[JudgeAnalytics],
) -> String {
    let items: Vec<JsonRankedItem> = rankings
        .iter()
        .enumerate()
        .map(|(i, r)| JsonRankedItem {
            rank: i + 1,
            id: r.item,
            name: names[r.item as usize].clone(),
            score: r.score,
            lower_bound: r.lower_bound,
            upper_bound: r.upper_bound,
        })
        .collect();

    // Top-level bias: weighted average across judges by num_comparisons (panel-wide aggregate).
    let total_n: usize = judge_analytics.iter().map(|ja| ja.num_comparisons).sum();
    let (bias, bias_ci_low, bias_ci_high) = if total_n > 0 {
        let w = |x: f64, n: usize| x * (n as f64) / (total_n as f64);
        (
            judge_analytics.iter().map(|ja| w(ja.positional_bias, ja.num_comparisons)).sum(),
            judge_analytics.iter().map(|ja| w(ja.positional_bias_ci.0, ja.num_comparisons)).sum(),
            judge_analytics.iter().map(|ja| w(ja.positional_bias_ci.1, ja.num_comparisons)).sum(),
        )
    } else {
        (0.5, 0.5, 0.5)
    };

    let judge_analytics_json: Vec<JsonJudgeAnalytics> = judge_analytics
        .iter()
        .map(|ja| JsonJudgeAnalytics {
            judge_id: ja.judge_id,
            positional_bias: ja.positional_bias,
            positional_bias_ci_low: ja.positional_bias_ci.0,
            positional_bias_ci_high: ja.positional_bias_ci.1,
            decisiveness: ja.decisiveness,
            decisiveness_ci_low: ja.decisiveness_ci.map(|c| c.0),
            decisiveness_ci_high: ja.decisiveness_ci.map(|c| c.1),
            num_comparisons: ja.num_comparisons,
        })
        .collect();

    let output = JsonOutput {
        items,
        total_comparisons,
        rounds,
        positional_bias: bias,
        positional_bias_ci_low: bias_ci_low,
        positional_bias_ci_high: bias_ci_high,
        judge_analytics: judge_analytics_json,
    };

    serde_json::to_string_pretty(&output).unwrap()
}

/// Print results as JSON.
pub fn print_json(
    rankings: &[RankedItem],
    names: &[String],
    rounds: usize,
    total_comparisons: usize,
    judge_analytics: &[JudgeAnalytics],
) {
    println!(
        "{}",
        build_json(rankings, names, rounds, total_comparisons, judge_analytics)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rankings() -> (Vec<RankedItem>, Vec<String>) {
        let rankings = vec![
            RankedItem {
                item: 2,
                score: 1.58,
                lower_bound: 1.20,
                upper_bound: 1.97,
            },
            RankedItem {
                item: 0,
                score: 0.75,
                lower_bound: 0.45,
                upper_bound: 1.05,
            },
            RankedItem {
                item: 1,
                score: 0.42,
                lower_bound: 0.12,
                upper_bound: 0.68,
            },
        ];
        let names = vec![
            "Apple".to_string(),
            "Banana".to_string(),
            "Mango".to_string(),
        ];
        (rankings, names)
    }

    fn sample_analytics() -> Vec<JudgeAnalytics> {
        vec![JudgeAnalytics {
            judge_id: 42,
            positional_bias: 0.523,
            positional_bias_ci: (0.481, 0.567),
            decisiveness: Some(1.0),
            decisiveness_ci: Some((0.9, 1.1)),
            num_comparisons: 30,
        }]
    }

    #[test]
    fn test_json_contains_all_fields() {
        let (rankings, names) = sample_rankings();
        let json = build_json(&rankings, &names, 10, 30, &sample_analytics());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["rounds"], 10);
        assert_eq!(parsed["total_comparisons"], 30);
        assert_eq!(parsed["positional_bias"], 0.523);
        assert_eq!(parsed["positional_bias_ci_low"], 0.481);
        assert_eq!(parsed["positional_bias_ci_high"], 0.567);
    }

    #[test]
    fn test_json_items_structure() {
        let (rankings, names) = sample_rankings();
        let json = build_json(&rankings, &names, 10, 30, &sample_analytics());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        let items = parsed["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);

        assert_eq!(items[0]["rank"], 1);
        assert_eq!(items[0]["id"], 2);
        assert_eq!(items[0]["name"], "Mango");
        assert_eq!(items[0]["score"], 1.58);
        assert_eq!(items[0]["lower_bound"], 1.20);
        assert_eq!(items[0]["upper_bound"], 1.97);

        assert_eq!(items[2]["rank"], 3);
        assert_eq!(items[2]["name"], "Banana");
    }

    #[test]
    fn test_json_is_valid() {
        let (rankings, names) = sample_rankings();
        let json = build_json(&rankings, &names, 5, 15, &sample_analytics());
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1234567), "1,234,567");
    }
}
