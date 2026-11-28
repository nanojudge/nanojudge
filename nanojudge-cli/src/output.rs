/// Output formatting: terminal table and JSON.
use crate::args::OutputFormat;
use crate::report::Reporter;
use nanojudge_core::{JudgeAnalytics, RankedItem, ScoringResult};
use serde::Serialize;
use std::collections::HashMap;

/// Structured result of `rank` or `score`, ready for programmatic inspection
/// or rendering via [`print`], [`print_table`], [`print_json`], or [`to_json`].
#[derive(Debug, Clone)]
pub struct RankingOutput {
    pub scoring: ScoringResult,
    pub item_names: Vec<String>,
    pub item_hashes: Vec<String>,
    pub edge_counts: Vec<usize>,
    /// Every judgement the ranking is built on, including `loaded_judgements`.
    pub total_judgements: usize,
    /// Judgements from `--load-judgements` that contributed to this ranking.
    pub loaded_judgements: Option<usize>,
    pub confidence_level: f64,
    pub judge_names: HashMap<u64, String>,
    pub judge_tokens: HashMap<u64, (u64, u64)>,
    pub judge_avg_wall_time: HashMap<u64, f64>,
    /// Per-refit interim rankings when `--emit-interim-rankings` is set.
    pub interim_rankings: Option<Vec<(usize, Vec<String>)>>,
}

#[derive(Serialize)]
struct JsonRankedItem {
    rank: usize,
    name: String,
    /// Truncated SHA-256 of the item text (same value as `item1_text_hash` /
    /// `item2_text_hash` in the judgement logs). The stable identity of the
    /// item; `name` is a display title and is not unique.
    text_hash: String,
    score: f64,
    lower_bound: f64,
    upper_bound: f64,
    edges: usize,
}

#[derive(Serialize)]
struct JsonJudgeAnalytics {
    judge_id: u64,
    positional_bias: f64,
    positional_bias_ci_low: f64,
    positional_bias_ci_high: f64,
    num_edges: usize,
}

/// One refit's interim ranking, emitted only when `--emit-interim-rankings` is
/// set. `order` lists item names best-to-worst after that refit; `judgements`
/// is the cumulative successful-judgement count through that refit, including
/// any loaded judgements.
#[derive(Serialize)]
struct JsonInterimRanking {
    judgements: usize,
    order: Vec<String>,
}

#[derive(Serialize)]
struct JsonOutput {
    items: Vec<JsonRankedItem>,
    /// Every judgement the ranking is built on, including `loaded_judgements`.
    total_judgements: usize,
    /// Judgements from `--load-judgements` that contributed to this ranking.
    /// Absent when nothing was loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded_judgements: Option<usize>,
    positional_bias: f64,
    positional_bias_ci_low: f64,
    positional_bias_ci_high: f64,
    judge_analytics: Vec<JsonJudgeAnalytics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interim_rankings: Option<Vec<JsonInterimRanking>>,
}

/// Format a judgement count, e.g. "30 judgements, including 24 loaded".
/// `total` already includes `loaded`; with nothing loaded it is "6 judgements".
pub fn judgement_count(total: usize, loaded: Option<usize>, noun: &str) -> String {
    match loaded {
        Some(loaded) => format!("{total} {noun}, including {loaded} loaded"),
        None => format!("{total} {noun}"),
    }
}

/// Default output format: table when stdout is a terminal, JSON otherwise.
pub fn default_format() -> OutputFormat {
    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        OutputFormat::Table
    } else {
        OutputFormat::Json
    }
}

/// Print results in the requested output format.
pub fn print(out: &RankingOutput, format: OutputFormat) {
    match format {
        OutputFormat::Json => print_json(out),
        OutputFormat::Table => print_table(out),
    }
}

/// Print results as a formatted terminal table.
pub fn print_table(out: &RankingOutput) {
    let rankings = &out.scoring.rankings;
    let names = &out.item_names;
    let edge_counts = &out.edge_counts;
    let judge_analytics = &out.scoring.judge_analytics;
    let ci_label = format!("{:.0}% CI", out.confidence_level * 100.0);

    // Find the widest item name for padding
    let name_width = rankings
        .iter()
        .map(|r| names[r.item as usize].chars().count())
        .max()
        .unwrap_or(4)
        .max(4); // at least "Item"

    // Header
    println!(
        " # | {:<name_width$} |   Score | {:>10} | {:>11} | Edges",
        "Item",
        format!("{} Low", ci_label),
        format!("{} High", ci_label),
    );
    println!(
        "---|-{}-|---------|------------|-------------|------",
        "-".repeat(name_width)
    );

    // Rows
    for (i, r) in rankings.iter().enumerate() {
        let name = &names[r.item as usize];
        let edges = edge_counts[r.item as usize];
        println!(
            "{:>2} | {:<name_width$} | {:>7.4} | {:>10.2} | {:>11.2} | {:>5}",
            i + 1,
            name,
            r.score,
            r.lower_bound,
            r.upper_bound,
            edges,
        );
    }

    println!(
        "\n{} items ranked ({})",
        rankings.len(),
        judgement_count(out.total_judgements, out.loaded_judgements, "judgements"),
    );

    // Print per-judge analytics
    if judge_analytics.len() == 1 {
        let ja = &judge_analytics[0];
        println!(
            "Position bias — estimated: {:.3} [{:.3}, {:.3}] (corrected for in scores, 0.5 = no bias)",
            ja.positional_bias, ja.positional_bias_ci.0, ja.positional_bias_ci.1,
        );
        if let Some(&(input, output)) = out.judge_tokens.get(&ja.judge_id)
            && (input > 0 || output > 0)
        {
            println!(
                "Tokens — input: {}, output: {}",
                format_count(input as usize),
                format_count(output as usize)
            );
        }
        if let Some(&avg) = out.judge_avg_wall_time.get(&ja.judge_id)
            && avg > 0.0
        {
            println!("Avg collection time: {}", format_duration(avg));
        }
    } else {
        print_judge_panel_analytics(
            judge_analytics,
            &out.judge_names,
            &out.judge_tokens,
            &out.judge_avg_wall_time,
        );
        println!(
            "Panel position bias — estimated: {:.3} [{:.3}, {:.3}] (corrected for in scores, 0.5 = no bias)",
            out.scoring.panel_positional_bias,
            out.scoring.panel_positional_bias_ci.0,
            out.scoring.panel_positional_bias_ci.1,
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
        .map(|ja| judge_names.get(&ja.judge_id).map_or(16, |n| n.chars().count()))
        .max()
        .unwrap_or(5)
        .max(5); // at least "Judge"

    println!();

    // Header
    let mut header = format!(
        "  {:<name_width$}   {:>11}   {:>15}",
        "Judge", "Edges", "Bias (->item1)"
    );
    let mut separator = format!(
        "  {:<name_width$}   {:>11}   {:>15}",
        "\u{2500}".repeat(name_width.min(30)),
        "\u{2500}".repeat(11),
        "\u{2500}".repeat(15)
    );

    if has_tokens {
        header += &format!("   {:>13}   {:>13}", "Input tokens", "Output tokens");
        separator += &format!(
            "   {:>13}   {:>13}",
            "\u{2500}".repeat(13),
            "\u{2500}".repeat(13)
        );
    }
    if has_wall_time {
        header += &format!("   {:>14}", "Avg collection");
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
            format_count(ja.num_edges),
            bias_str,
        );

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
}

/// Report a compact live ranking table after a refit.
/// If `limit` is 0, emits all items.
pub fn print_live_table(
    rankings: &[RankedItem],
    names: &[String],
    refit: usize,
    total_judgements: usize,
    loaded_judgements: Option<usize>,
    limit: usize,
    reporter: &dyn Reporter,
) {
    let show = if limit == 0 || limit >= rankings.len() {
        rankings
    } else {
        &rankings[..limit]
    };

    let name_width = show
        .iter()
        .map(|r| names[r.item as usize].chars().count())
        .max()
        .unwrap_or(4)
        .max(4);

    reporter.line(&format!(
        "\n── Refit {} ({}) ──",
        refit,
        judgement_count(total_judgements, loaded_judgements, "judgements"),
    ));
    for (i, r) in show.iter().enumerate() {
        reporter.line(&format!(
            " {:>2}. {:<name_width$}  {:>7.4}  [{:.2}, {:.2}]",
            i + 1,
            names[r.item as usize],
            r.score,
            r.lower_bound,
            r.upper_bound,
        ));
    }
    if limit > 0 && limit < rankings.len() {
        reporter.line(&format!("     … and {} more", rankings.len() - limit));
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

/// Format results as pretty-printed JSON.
pub fn to_json(out: &RankingOutput) -> String {
    let items: Vec<JsonRankedItem> = out
        .scoring
        .rankings
        .iter()
        .enumerate()
        .map(|(i, r)| JsonRankedItem {
            rank: i + 1,
            name: out.item_names[r.item as usize].clone(),
            text_hash: out.item_hashes[r.item as usize].clone(),
            score: r.score,
            lower_bound: r.lower_bound,
            upper_bound: r.upper_bound,
            edges: out.edge_counts[r.item as usize],
        })
        .collect();

    let judge_analytics_json: Vec<JsonJudgeAnalytics> = out
        .scoring
        .judge_analytics
        .iter()
        .map(|ja| JsonJudgeAnalytics {
            judge_id: ja.judge_id,
            positional_bias: ja.positional_bias,
            positional_bias_ci_low: ja.positional_bias_ci.0,
            positional_bias_ci_high: ja.positional_bias_ci.1,
            num_edges: ja.num_edges,
        })
        .collect();

    let interim_rankings_json = out.interim_rankings.as_ref().map(|refits| {
        refits
            .iter()
            .map(|(judgements, order)| JsonInterimRanking {
                judgements: *judgements,
                order: order.clone(),
            })
            .collect()
    });

    let output = JsonOutput {
        items,
        total_judgements: out.total_judgements,
        loaded_judgements: out.loaded_judgements,
        positional_bias: out.scoring.panel_positional_bias,
        positional_bias_ci_low: out.scoring.panel_positional_bias_ci.0,
        positional_bias_ci_high: out.scoring.panel_positional_bias_ci.1,
        judge_analytics: judge_analytics_json,
        interim_rankings: interim_rankings_json,
    };

    serde_json::to_string_pretty(&output).unwrap()
}

/// Print results as JSON.
pub fn print_json(out: &RankingOutput) {
    println!("{}", to_json(out));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_output(total_judgements: usize, loaded_judgements: Option<usize>) -> RankingOutput {
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
        RankingOutput {
            scoring: ScoringResult {
                rankings,
                selection_weights: None,
                partition_log_confidence: None,
                item_means: vec![0.75, 0.42, 1.58],
                item_stds: vec![0.15, 0.14, 0.20],
                judge_analytics: vec![JudgeAnalytics {
                    judge_id: 42,
                    positional_bias: 0.523,
                    positional_bias_ci: (0.481, 0.567),
                    num_edges: 30,
                }],
                panel_positional_bias: 0.523,
                panel_positional_bias_ci: (0.481, 0.567),
            },
            item_names: vec!["Apple".to_string(), "Banana".to_string(), "Mango".to_string()],
            item_hashes: vec![
                "aaaaaaaaaaaaaaaa".into(),
                "bbbbbbbbbbbbbbbb".into(),
                "cccccccccccccccc".into(),
            ],
            edge_counts: vec![11, 22, 33],
            total_judgements,
            loaded_judgements,
            confidence_level: 0.95,
            judge_names: HashMap::new(),
            judge_tokens: HashMap::new(),
            judge_avg_wall_time: HashMap::new(),
            interim_rankings: None,
        }
    }

    #[test]
    fn test_json_items_carry_text_hash() {
        let out = sample_output(30, None);
        let json = to_json(&out);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let items = parsed["items"].as_array().unwrap();
        for it in items {
            let idx = out.item_names.iter().position(|n| n == it["name"].as_str().unwrap()).unwrap();
            assert_eq!(it["text_hash"], out.item_hashes[idx]);
        }
    }

    #[test]
    fn test_json_contains_all_fields() {
        let out = sample_output(30, None);
        let json = to_json(&out);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["total_judgements"], 30);
        assert!(parsed.get("loaded_judgements").is_none());
        assert_eq!(parsed["positional_bias"], 0.523);
        assert_eq!(parsed["positional_bias_ci_low"], 0.481);
        assert_eq!(parsed["positional_bias_ci_high"], 0.567);
    }

    #[test]
    fn test_json_loaded_judgements() {
        let out = sample_output(30, Some(24));
        let json = to_json(&out);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["total_judgements"], 30);
        assert_eq!(parsed["loaded_judgements"], 24);
    }

    #[test]
    fn test_judgement_count() {
        assert_eq!(judgement_count(6, None, "judgements"), "6 judgements");
        assert_eq!(judgement_count(30, Some(24), "judgements"), "30 judgements, including 24 loaded");
        assert_eq!(judgement_count(30, Some(24), "lineup judgements"), "30 lineup judgements, including 24 loaded");
    }

    #[test]
    fn test_json_items_structure() {
        let out = sample_output(30, None);
        let json = to_json(&out);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        let items = parsed["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);

        assert_eq!(items[0]["rank"], 1);
        assert_eq!(items[0]["name"], "Mango");
        assert_eq!(items[0]["score"], 1.58);
        assert_eq!(items[0]["lower_bound"], 1.20);
        assert_eq!(items[0]["upper_bound"], 1.97);
        assert_eq!(items[0]["edges"], 33);

        assert_eq!(items[2]["rank"], 3);
        assert_eq!(items[2]["name"], "Banana");
    }

    #[test]
    fn test_json_is_valid() {
        let out = sample_output(15, None);
        let json = to_json(&out);
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
