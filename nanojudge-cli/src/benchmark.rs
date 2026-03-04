/// Benchmark command: measures endpoint throughput, latency, reliability, and positional bias.
///
/// Runs N pairs of comparisons, each in both directions (A vs B and B vs A),
/// collecting timing, token usage, verdict distribution, and flip rate statistics.

use crate::bail;
use crate::llm::{LlmConfig, send_comparison_request};
use crate::prompt::build_prompt;
use rand::Rng;
use rand::seq::SliceRandom;
use reqwest::Client;
use serde::Deserialize;
use std::io::Write;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

#[derive(Deserialize)]
struct BenchmarkQuestion {
    question: String,
    target_length: String,
    item1: String,
    item2: String,
}

const BENCHMARK_QUESTIONS: &str = include_str!("benchmark_questions.json");

/// Result of a single benchmark comparison (one direction).
struct SingleResult {
    latency_secs: f64,
    prompt_tokens: u64,
    completion_tokens: u64,
    response_len_chars: usize,
    verdict_letter: Option<usize>, // 0=A, 1=B, 2=C, 3=D, 4=E
    parseable: bool,
    http_error: bool,
    prompt_text: String,
    response_text: String,
}

/// Run the benchmark.
pub async fn run_benchmark(
    endpoint: &str,
    model: &str,
    api_key: Option<String>,
    num_pairs: usize,
    concurrency: usize,
    temperature: f64,
    temperature_jitter: f64,
    presence_penalty: Option<f64>,
    top_p: Option<f64>,
    narrow_win: f64,
    template: &str,
) {
    let questions: Vec<BenchmarkQuestion> = serde_json::from_str(BENCHMARK_QUESTIONS)
        .unwrap_or_else(|e| bail(format!("Failed to parse embedded benchmark questions: {e}")));

    // Sample N questions (with replacement if num_pairs > questions.len())
    let mut rng = rand::rng();
    let selected: Vec<&BenchmarkQuestion> = if num_pairs <= questions.len() {
        let mut indices: Vec<usize> = (0..questions.len()).collect();
        indices.shuffle(&mut rng);
        indices[..num_pairs].iter().map(|&i| &questions[i]).collect()
    } else {
        (0..num_pairs)
            .map(|_| {
                let idx = rng.random_range(0..questions.len());
                &questions[idx]
            })
            .collect()
    };

    let total_comparisons = num_pairs * 2; // Each pair runs both directions
    eprintln!(
        "Running benchmark: {} pairs x 2 directions = {} comparisons (concurrency: {})",
        num_pairs, total_comparisons, concurrency
    );
    eprintln!("Endpoint: {} | Model: {}", endpoint, model);

    let llm_config = Arc::new(LlmConfig {
        endpoint: endpoint.to_string(),
        model: model.to_string(),
        api_key,
        temperature,
        temperature_jitter,
        presence_penalty,
        top_p,
    });

    let client = Client::new();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));

    // Build all comparison tasks: for each pair, run forward (item1 vs item2) and reverse (item2 vs item1)
    struct ComparisonTask {
        pair_index: usize,
        is_forward: bool,
        prompt: String,
    }

    let mut tasks: Vec<ComparisonTask> = Vec::with_capacity(total_comparisons);
    for (i, q) in selected.iter().enumerate() {
        let forward_prompt = build_prompt(template, &q.question, &q.item1, &q.item2, &q.target_length);
        let reverse_prompt = build_prompt(template, &q.question, &q.item2, &q.item1, &q.target_length);
        tasks.push(ComparisonTask { pair_index: i, is_forward: true, prompt: forward_prompt });
        tasks.push(ComparisonTask { pair_index: i, is_forward: false, prompt: reverse_prompt });
    }

    // Shuffle to avoid sequential bias
    tasks.shuffle(&mut rng);

    let wall_clock_start = Instant::now();

    // Run all comparisons with bounded concurrency
    let mut handles = Vec::with_capacity(tasks.len());
    for task in tasks {
        let sem = semaphore.clone();
        let client = client.clone();
        let config = llm_config.clone();
        let prompt = task.prompt;
        let pair_index = task.pair_index;
        let is_forward = task.is_forward;

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let start = Instant::now();
            let result = send_comparison_request(&client, &config, &prompt, narrow_win).await;
            let latency = start.elapsed().as_secs_f64();

            let single = match result {
                Ok((parse_result, content, usage)) => {
                    let (prompt_tokens, completion_tokens) = match usage {
                        Some(u) => (u.prompt_tokens, u.completion_tokens),
                        None => (0, 0),
                    };
                    SingleResult {
                        latency_secs: latency,
                        prompt_tokens,
                        completion_tokens,
                        response_len_chars: content.len(),
                        verdict_letter: parse_result.item1_win_probability.and_then(|p| {
                            // Map probability back to verdict letter index
                            if (p - 1.0).abs() < 0.01 { Some(0) }      // A
                            else if (p - 0.8).abs() < 0.01 { Some(1) }  // B (approximate)
                            else if (p - 0.5).abs() < 0.01 { Some(2) }  // C
                            else if (p - 0.2).abs() < 0.01 { Some(3) }  // D (approximate)
                            else if p.abs() < 0.01 { Some(4) }          // E
                            else { None } // logprob-derived, not a clean letter
                        }),
                        parseable: parse_result.item1_win_probability.is_some(),
                        http_error: false,
                        prompt_text: prompt,
                        response_text: content,
                    }
                }
                Err(ref e) => {
                    eprintln!("\n  Error: {e}");
                    SingleResult {
                    latency_secs: latency,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    response_len_chars: 0,
                    verdict_letter: None,
                    parseable: false,
                    http_error: true,
                    prompt_text: prompt,
                    response_text: format!("[HTTP ERROR: {e}]"),
                }}
            };

            // Print progress dot
            eprint!(".");

            (pair_index, is_forward, single)
        });

        handles.push(handle);
    }

    // Collect results
    let mut forward_results: Vec<Option<SingleResult>> = (0..num_pairs).map(|_| None).collect();
    let mut reverse_results: Vec<Option<SingleResult>> = (0..num_pairs).map(|_| None).collect();

    for handle in handles {
        let (pair_index, is_forward, result) = handle.await.unwrap();
        if is_forward {
            forward_results[pair_index] = Some(result);
        } else {
            reverse_results[pair_index] = Some(result);
        }
    }

    let wall_clock_secs = wall_clock_start.elapsed().as_secs_f64();
    eprintln!(" done ({:.1}s)\n", wall_clock_secs);

    // Flatten all results for aggregate stats
    let all_results: Vec<&SingleResult> = forward_results.iter()
        .chain(reverse_results.iter())
        .filter_map(|r| r.as_ref())
        .collect();

    // --- Compute stats ---

    // Throughput
    let successful: Vec<&&SingleResult> = all_results.iter().filter(|r| !r.http_error).collect();
    let total_prompt_tokens: u64 = successful.iter().map(|r| r.prompt_tokens).sum();
    let total_completion_tokens: u64 = successful.iter().map(|r| r.completion_tokens).sum();
    let total_tokens = total_prompt_tokens + total_completion_tokens;
    let has_token_data = total_tokens > 0;

    let comparisons_per_sec = successful.len() as f64 / wall_clock_secs;

    // Latency
    let mut latencies: Vec<f64> = successful.iter().map(|r| r.latency_secs).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentile = |sorted: &[f64], p: f64| -> f64 {
        if sorted.is_empty() { return 0.0; }
        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    // Reliability
    let http_errors = all_results.iter().filter(|r| r.http_error).count();
    let parseable_count = all_results.iter().filter(|r| r.parseable).count();
    let non_http_count = all_results.len() - http_errors;

    // Verdict distribution (A-E)
    let verdict_labels = ["A (Option 1 clearly wins)", "B (Option 1 narrowly wins)", "C (Draw)", "D (Option 2 narrowly wins)", "E (Option 2 clearly wins)"];
    let mut verdict_counts = [0usize; 5];
    let mut option1_wins = 0usize;
    let mut option2_wins = 0usize;
    for r in &all_results {
        if let Some(letter) = r.verdict_letter {
            if letter < 5 { verdict_counts[letter] += 1; }
            match letter {
                0 | 1 => option1_wins += 1,
                3 | 4 => option2_wins += 1,
                _ => {}
            }
        }
    }
    let verdicts_with_winner = option1_wins + option2_wins;

    // Average response length
    let avg_response_chars: f64 = if !successful.is_empty() {
        successful.iter().map(|r| r.response_len_chars as f64).sum::<f64>() / successful.len() as f64
    } else {
        0.0
    };

    // --- Print report ---

    println!("── Throughput ──────────────────────────────────");
    println!("Comparisons/sec:       {:.2}", comparisons_per_sec);
    if has_token_data {
        println!("Output tokens/sec:     {:.0}", total_completion_tokens as f64 / wall_clock_secs);
        println!("Avg input tokens:      {:.0}", total_prompt_tokens as f64 / successful.len().max(1) as f64);
        println!("Avg output tokens:     {:.0}", total_completion_tokens as f64 / successful.len().max(1) as f64);
    } else {
        println!("(Token data not available from this endpoint)");
    }

    println!();
    println!("── Latency (per comparison) ────────────────────");
    if !latencies.is_empty() {
        println!("P50:    {:.2}s", percentile(&latencies, 0.50));
        println!("P95:    {:.2}s", percentile(&latencies, 0.95));
        println!("P99:    {:.2}s", percentile(&latencies, 0.99));
        println!("Min:    {:.2}s    Max: {:.2}s", latencies.first().unwrap(), latencies.last().unwrap());
    } else {
        println!("(No successful comparisons)");
    }

    println!();
    println!("── Reliability ─────────────────────────────────");
    if non_http_count > 0 {
        println!("Verdict parse rate:    {:.1}%  ({}/{})", parseable_count as f64 / non_http_count as f64 * 100.0, parseable_count, non_http_count);
    }
    println!("HTTP error rate:       {:.1}%  ({}/{})", http_errors as f64 / all_results.len() as f64 * 100.0, http_errors, all_results.len());
    println!("Avg response length:   {:.0} chars", avg_response_chars);

    println!();
    println!("── Verdict Distribution ────────────────────────");
    let total_verdicts: usize = verdict_counts.iter().sum();
    if total_verdicts > 0 {
        for (i, label) in verdict_labels.iter().enumerate() {
            let pct = verdict_counts[i] as f64 / total_verdicts as f64 * 100.0;
            println!("{}: {:>5.1}%  ({})", label, pct, verdict_counts[i]);
        }
    } else {
        println!("(No parseable verdicts)");
    }

    println!();
    println!("── Positional Bias ─────────────────────────────");
    if verdicts_with_winner > 0 {
        println!("Option 1 win rate:     {:.1}%", option1_wins as f64 / verdicts_with_winner as f64 * 100.0);
    } else {
        println!("(Insufficient data for positional bias)");
    }

    if has_token_data {
        println!();
        println!("── Cost Estimate ───────────────────────────────");
        println!("Tokens used:           {:>7}  ({} in + {} out)", total_tokens, total_prompt_tokens, total_completion_tokens);
        if successful.len() > 1 {
            let tokens_per_comparison = total_tokens as f64 / successful.len() as f64;
            println!("At this rate, 1000 comparisons ≈ {:.0}K tokens", tokens_per_comparison * 1000.0 / 1000.0);
        }
    }

    // --- Write detailed log file ---
    let timestamp = {
        let secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // Manual UTC formatting (avoids chrono dependency)
        let s = secs % 60;
        let m = (secs / 60) % 60;
        let h = (secs / 3600) % 24;
        let days = secs / 86400;
        // Days since epoch to Y-M-D (simplified)
        let mut y = 1970i64;
        let mut remaining = days as i64;
        loop {
            let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
            if remaining < days_in_year { break; }
            remaining -= days_in_year;
            y += 1;
        }
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut mo = 0;
        for (i, &md) in month_days.iter().enumerate() {
            if remaining < md as i64 { mo = i + 1; break; }
            remaining -= md as i64;
        }
        let d = remaining + 1;
        format!("{:04}-{:02}-{:02}_{:02}{:02}{:02}", y, mo, d, h, m, s)
    };

    let log_filename = format!("benchmark_{}.log", timestamp);
    let verdict_names = ["A", "B", "C", "D", "E"];

    let mut file = std::fs::File::create(&log_filename)
        .unwrap_or_else(|e| bail(format!("Failed to create log file {log_filename}: {e}")));

    writeln!(file, "NanoJudge Benchmark Log").unwrap();
    writeln!(file, "Generated: {} UTC", timestamp.replace('_', " ")).unwrap();
    writeln!(file, "Endpoint: {} | Model: {}", endpoint, model).unwrap();
    writeln!(file, "Pairs: {} | Comparisons: {} | Wall time: {:.1}s", num_pairs, total_comparisons, wall_clock_secs).unwrap();
    writeln!(file, "Parse rate: {}/{} ({:.1}%)", parseable_count, non_http_count, parseable_count as f64 / non_http_count.max(1) as f64 * 100.0).unwrap();
    writeln!(file, "").unwrap();

    let mut comparison_num = 0usize;
    for i in 0..num_pairs {
        for (direction, results) in [("FORWARD", &forward_results), ("REVERSE", &reverse_results)] {
            if let Some(ref result) = results[i] {
                comparison_num += 1;
                let status = if result.http_error {
                    "HTTP ERROR".to_string()
                } else if result.parseable {
                    match result.verdict_letter {
                        Some(v) if v < 5 => format!("PARSED (Verdict: {})", verdict_names[v]),
                        _ => "PARSED (logprob-derived)".to_string(),
                    }
                } else {
                    "FAILED TO PARSE".to_string()
                };

                writeln!(file, "{}", "=".repeat(80)).unwrap();
                writeln!(file, "#{} | Pair {} {} | {:.2}s | {}", comparison_num, i + 1, direction, result.latency_secs, status).unwrap();
                writeln!(file, "{}", "=".repeat(80)).unwrap();
                writeln!(file, "").unwrap();
                writeln!(file, "--- PROMPT ---").unwrap();
                writeln!(file, "{}", result.prompt_text).unwrap();
                writeln!(file, "").unwrap();
                writeln!(file, "--- RESPONSE ---").unwrap();
                writeln!(file, "{}", result.response_text).unwrap();
                writeln!(file, "").unwrap();
            }
        }
    }

    println!();
    println!("Detailed log written to: {}", log_filename);
}
