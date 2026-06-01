use nanojudge_core::{
    ComparisonInput, EngineConfig, JudgeInfo, RankingEngine, ScoringOptions, Strategy,
    calculate_pairs_for_round, calculate_rounds_for_target_comparisons,
    calculate_total_expected_comparisons, run_scoring,
};
use rand::seq::SliceRandom;
use reqwest::Client;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::args::RankArgs;
use crate::bail;
use crate::config;
use crate::items::load_items;
use crate::llm::{LlmConfig, compare_pair};
use crate::output;
use crate::resolve::{resolve_config, resolve_judges};

#[derive(Default)]
struct JudgeStats {
    input_tokens: u64,
    output_tokens: u64,
    max_tokens_hits: usize,
    total_responses: usize,
    wall_time_sum: f64,
    round_count: usize,
}

fn resolve_save_path(path: &Path, prefix: &str) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    if path.is_dir() {
        path.join(format!("{prefix}-{ts}.jsonl"))
    } else {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if !parent.exists() {
            bail(format!("Directory {} does not exist", parent.display()));
        }
        path.to_path_buf()
    }
}

/// Parse a criterion file into a list of criteria, splitting on ---CRITERION---.
fn parse_criteria(content: &str) -> Vec<String> {
    content
        .split("---CRITERION---")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Assign pairs to judges for one round, balancing cumulative usage across rounds.
///
/// `cumulative_total` is the total pairs INCLUDING this round. Each judge's target
/// is `cumulative_total * weight`, minus what they've already been assigned. This
/// ensures even distribution over time rather than independent per-round allocation.
/// Updates `cumulative_assigned` in place.
fn assign_pairs_to_judges(
    round_pairs: usize,
    normalized_weights: &[f64],
    cumulative_assigned: &mut [usize],
    cumulative_total: usize,
    rng: &mut impl rand::Rng,
) -> Vec<usize> {
    let num_judges = normalized_weights.len();

    let mut counts: Vec<usize> = Vec::with_capacity(num_judges);
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(num_judges);
    let mut assigned = 0usize;

    for (i, &w) in normalized_weights.iter().enumerate() {
        let target_this_round = (w * cumulative_total as f64) - cumulative_assigned[i] as f64;
        let floor = (target_this_round.floor() as usize).min(round_pairs.saturating_sub(assigned));
        counts.push(floor);
        remainders.push((i, target_this_round - floor as f64));
        assigned += floor;
    }

    remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(judge_idx, _) in remainders.iter().take(round_pairs - assigned) {
        counts[judge_idx] += 1;
    }

    for (i, &count) in counts.iter().enumerate() {
        cumulative_assigned[i] += count;
    }

    let mut assignments: Vec<usize> = Vec::with_capacity(round_pairs);
    for (judge_idx, &count) in counts.iter().enumerate() {
        assignments.extend(std::iter::repeat_n(judge_idx, count));
    }
    assignments.shuffle(rng);

    assignments
}

pub async fn run(args: RankArgs) {
    let config_path = args.config.clone().unwrap_or_else(config::config_path);
    let cfg = config::load_config(&config_path);
    let resolved = resolve_config(&args.cfg, &cfg);

    // Resolve judges from [[judge]] blocks
    let judges = resolve_judges(&args.cfg, &cfg, &config_path, resolved.reasoning_enabled);
    let logprobs_mode = judges[0].logprobs;

    if !logprobs_mode {
        eprintln!("Warning: Running without logprobs. Requires more comparisons to reach equivalent accuracy as when using logprobs.");
    }

    let (titles, texts) = load_items(&args);
    let item_ids: Vec<i64> = (0..texts.len() as i64).collect();

    let rounds = if let Some(target) = resolved.comparisons {
        let pairs_per_round = calculate_pairs_for_round(texts.len());
        if pairs_per_round == 0 {
            bail("Cannot run comparisons: need at least 2 items.");
        }
        let r = calculate_rounds_for_target_comparisons(texts.len(), target);
        if r == 0 {
            bail(format!(
                "--comparisons ({target}) is less than one round ({pairs_per_round} comparisons). Use at least {pairs_per_round}.",
            ));
        }
        let actual = calculate_total_expected_comparisons(texts.len(), r);
        if actual != target {
            eprintln!(
                "Running {} comparisons instead of {}, to align with round boundaries ({} per round).",
                actual, target, pairs_per_round,
            );
        }
        r
    } else {
        resolved.rounds.unwrap_or_else(|| {
            bail(format!("No rounds or comparisons specified. Pass --rounds, --comparisons, or set it in {}", config_path.display()));
        })
    };

    // Build JudgeInfo for the core engine
    let judge_ids: Vec<u64> = judges.iter().map(|j| j.judge_id).collect();
    let judge_info = JudgeInfo {
        judge_ids: judge_ids.clone(),
        logprobs_mode,
    };

    // Build per-judge LlmConfigs and semaphores
    let judge_llm_configs: Vec<Arc<LlmConfig>> = judges.iter().map(|j| {
        Arc::new(LlmConfig {
            endpoint: j.endpoint.clone(),
            model: j.model.clone(),
            api_key: j.api_key.clone(),
            temperature: j.temperature,
            temperature_jitter: j.temperature_jitter,
            presence_penalty: j.presence_penalty,
            top_p: j.top_p,
            logprobs: j.logprobs,
            max_tokens: j.max_tokens,
            reasoning_effort: j.reasoning_effort.clone(),
        })
    }).collect();

    let judge_semaphores: Vec<Arc<tokio::sync::Semaphore>> = judges.iter()
        .map(|j| Arc::new(tokio::sync::Semaphore::new(j.concurrency)))
        .collect();

    // Compute normalized weights for pair assignment
    let total_weight: f64 = judges.iter().map(|j| j.weight).sum();
    let normalized_weights: Vec<f64> = judges.iter().map(|j| j.weight / total_weight).collect();
    // Per-judge narrow_win values
    let judge_narrow_wins: Vec<f64> = judges.iter().map(|j| j.narrow_win).collect();
    // Per-judge min_logprob_coverage values
    let judge_min_logprob_coverages: Vec<f64> = judges.iter().map(|j| j.min_logprob_coverage).collect();

    let criteria: Vec<String> = if let Some(ref path) = args.criterion_file {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| bail(format!("Failed to read criterion file {}: {e}", path.display())));
        let parts = parse_criteria(&content);
        if parts.is_empty() {
            bail(format!("No criteria found in {}", path.display()));
        }
        parts
    } else {
        vec![args.criterion.clone().unwrap()]
    };

    let prompt_template = Arc::new(resolved.prompt_template.clone());

    let client = Client::new();
    let titles = Arc::new(titles);
    let texts = Arc::new(texts);

    let total_planned = calculate_total_expected_comparisons(texts.len(), rounds);

    if resolved.verbose {
        eprintln!(
            "Ranking {} items across {} rounds ({} comparisons planned)",
            texts.len(),
            rounds,
            total_planned,
        );
        if criteria.len() == 1 {
            eprintln!("Criterion: \"{}\"", criteria[0]);
        } else {
            eprintln!("Criteria ({} variants):", criteria.len());
            for (i, c) in criteria.iter().enumerate() {
                eprintln!("  {}: \"{}\"", i + 1, c);
            }
        }

        if judges.len() == 1 {
            eprintln!("Endpoint: {} | Model: {}", judges[0].endpoint, judges[0].model);
        } else {
            eprintln!("Judge panel ({} judges):", judges.len());
            for j in &judges {
                eprintln!(
                    "  {} — {} (concurrency: {}, weight: {:.0}%)",
                    j.display_name,
                    j.endpoint,
                    j.concurrency,
                    j.weight / total_weight * 100.0,
                );
            }
        }
    }

    // Set up comparison saving if requested
    let save_file = if let Some(ref save_path) = resolved.save_comparisons {
        let path = resolve_save_path(save_path, "comparisons");

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));

        if resolved.verbose {
            eprintln!("Saving comparisons to {}", path.display());
        }

        Some(std::sync::Mutex::new(file))
    } else {
        None
    };

    // Set up failure saving if requested
    let failures_file = if let Some(ref save_path) = resolved.save_failures {
        let path = resolve_save_path(save_path, "failures");

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));

        if resolved.verbose {
            eprintln!("Saving failures to {}", path.display());
        }

        Some(std::sync::Mutex::new(file))
    } else {
        None
    };

    let strategy = resolved.strategy;

    if resolved.top_k.is_some() && matches!(strategy, Strategy::Balanced) {
        eprintln!("Warning: --top-k has no effect with the balanced strategy. It only applies to --strategy top-heavy.");
    }

    // Pure heuristic — no empirical basis. Just a guess at how many top
    // positions users typically care about for a given list size.
    let top_k = resolved.top_k.unwrap_or_else(|| {
        ((texts.len() as f64).sqrt() * 3.0) as usize
    }).min(texts.len() - 1);

    let engine_config = EngineConfig {
        strategy,
        matchmaking_sharpness: resolved.matchmaking_sharpness,
        min_games_before_strategy: resolved.min_games_before_strategy,
        number_of_rounds: Some(rounds),
    };
    let mut engine = RankingEngine::new(&item_ids, engine_config);

    let analysis_length = resolved.analysis_length.clone();
    let max_retries = resolved.retries;

    // Judge display names (Arc for sharing across tasks)
    let judge_display_names: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.display_name.clone()).collect());
    let judge_models: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.model.clone()).collect());
    let judge_endpoints: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.endpoint.clone()).collect());

    let mut total_comparisons: usize = 0;
    let mut total_retries: usize = 0;
    let mut failed_http: usize = 0;
    let mut failed_parse: usize = 0;
    let mut judge_stats: Vec<JudgeStats> = (0..judges.len()).map(|_| JudgeStats::default()).collect();

    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let cancelled = cancelled.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            cancelled.store(true, Ordering::Relaxed);
        });
    }

    let mut cumulative_judge_pairs: Vec<usize> = vec![0; judges.len()];
    let mut cumulative_criterion_pairs: Vec<usize> = vec![0; criteria.len()];
    let mut cumulative_total_pairs: usize = 0;

    let mut interim_warm_start: Option<nanojudge_core::WarmStartState> = None;
    for round in 0..rounds {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let pairs = engine.generate_pairs_for_round(round);
        let round_start = std::time::Instant::now();

        if resolved.verbose {
            eprintln!("Round {}/{}: {} pairs", round + 1, rounds, pairs.len());
        }

        let mut rng = rand::rng();
        cumulative_total_pairs += pairs.len();
        let pair_assignments = assign_pairs_to_judges(
            pairs.len(),
            &normalized_weights,
            &mut cumulative_judge_pairs,
            cumulative_total_pairs,
            &mut rng,
        );

        let criterion_weights: Vec<f64> = vec![1.0 / criteria.len() as f64; criteria.len()];
        let criterion_assignments = assign_pairs_to_judges(
            pairs.len(),
            &criterion_weights,
            &mut cumulative_criterion_pairs,
            cumulative_total_pairs,
            &mut rng,
        );

        let mut handles = Vec::with_capacity(pairs.len());

        for (pair_idx, (id_a, id_b)) in pairs.iter().enumerate() {
            let judge_idx = pair_assignments[pair_idx];
            let sem = judge_semaphores[judge_idx].clone();
            let client = client.clone();
            let llm_config = judge_llm_configs[judge_idx].clone();
            let texts = texts.clone();
            let criterion = criteria[criterion_assignments[pair_idx]].clone();
            let analysis_length = analysis_length.clone();
            let template = prompt_template.clone();
            let id_a = *id_a;
            let id_b = *id_b;
            let narrow_win = judge_narrow_wins[judge_idx];
            let min_logprob_coverage = judge_min_logprob_coverages[judge_idx];
            let assigned_judge_id = judge_ids[judge_idx];
            let judge_name = judge_display_names[judge_idx].clone();

            let verbose = resolved.verbose;
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let result = compare_pair(
                    &client,
                    &llm_config,
                    &template,
                    &criterion,
                    &texts[id_a as usize],
                    &texts[id_b as usize],
                    id_a,
                    id_b,
                    narrow_win,
                    min_logprob_coverage,
                    &analysis_length,
                    max_retries,
                    verbose,
                    &judge_name,
                )
                .await;
                (result, assigned_judge_id, judge_idx, std::time::Instant::now())
            });

            handles.push((handle, judge_idx));
        }

        // Collect results
        let mut round_results: Vec<ComparisonInput> = Vec::new();
        let mut judge_last_finish: Vec<Option<std::time::Instant>> = vec![None; judges.len()];
        let mut judge_aborted: Vec<usize> = vec![0; judges.len()];

        for (handle, handle_judge_idx) in handles {
            if cancelled.load(Ordering::Relaxed) {
                handle.abort();
                judge_aborted[handle_judge_idx] += 1;
                continue;
            }
            let cancelled_ref = &cancelled;
            let result = tokio::select! {
                r = handle => r,
                _ = async { while !cancelled_ref.load(Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } } => {
                    judge_aborted[handle_judge_idx] += 1;
                    continue;
                }
            };
            match result {
                Ok((Ok(result), assigned_judge_id, judge_idx, finished_at)) => {
                    // Track latest finish time per judge for this round
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    total_retries += result.retries_used;
                    judge_stats[judge_idx].total_responses += 1;
                    if result.hit_max_tokens {
                        judge_stats[judge_idx].max_tokens_hits += 1;
                    }
                    if let Some(usage) = &result.usage {
                        judge_stats[judge_idx].input_tokens += usage.prompt_tokens;
                        judge_stats[judge_idx].output_tokens += usage.completion_tokens;
                    }
                    if let Some(p) = result.parse_result.item1_win_probability {
                        if let Some(ref file_mutex) = save_file {
                            let line = serde_json::json!({
                                "round": round + 1,
                                "item1": titles[result.item1_id as usize],
                                "item2": titles[result.item2_id as usize],
                                "probability": p,
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "response": result.response_text,
                            });
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }

                        round_results.push(ComparisonInput {
                            item1: result.item1_id,
                            item2: result.item2_id,
                            item1_win_probability: p,
                            judge_id: assigned_judge_id,
                        });
                    } else {
                        failed_parse += 1;
                        if let Some(ref file_mutex) = failures_file {
                            let line = serde_json::json!({
                                "round": round + 1,
                                "item1": titles[result.item1_id as usize],
                                "item2": titles[result.item2_id as usize],
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "response": result.response_text,
                            });
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }
                        if resolved.verbose {
                            if judges.len() > 1 {
                                eprintln!(
                                    "  Warning: unparseable response for {} vs {} [{}], skipping",
                                    titles[result.item1_id as usize],
                                    titles[result.item2_id as usize],
                                    judge_display_names[judge_idx],
                                );
                            } else {
                                eprintln!(
                                    "  Warning: unparseable response for {} vs {}, skipping",
                                    titles[result.item1_id as usize],
                                    titles[result.item2_id as usize],
                                );
                            }
                        }
                    }
                }
                Ok((Err(e), _judge_id, judge_idx, finished_at)) => {
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    failed_http += 1;
                    if resolved.verbose {
                        eprintln!(
                            "  Error [{}] (after exhausting {} retries): {e}",
                            judge_display_names[judge_idx], max_retries,
                        );
                    }
                }
                Err(e) => {
                    failed_http += 1;
                    if resolved.verbose {
                        eprintln!("  Task panicked: {e}");
                    }
                }
            }
        }

        if cancelled.load(Ordering::Relaxed) {
            // Print which judges had in-flight requests when cancelled
            for (i, judge) in judges.iter().enumerate() {
                if judge_aborted[i] > 0 {
                    eprintln!("  {} had {} in-flight requests", judge.display_name, judge_aborted[i]);
                }
            }
            break;
        }

        // Accumulate per-judge wall time for this round
        for (j, finish) in judge_last_finish.iter().enumerate() {
            if let Some(t) = finish {
                judge_stats[j].wall_time_sum += t.duration_since(round_start).as_secs_f64();
                judge_stats[j].round_count += 1;
            }
        }

        total_comparisons += round_results.len();

        let round_failed = pairs.len() - round_results.len();
        if resolved.verbose {
            eprintln!(
                "  Completed: {} successful, {} failed",
                round_results.len(),
                round_failed,
            );
        }

        if round_failed == pairs.len() {
            eprintln!(
                "Warning: all {} comparisons in round {} failed. \
                 If your endpoint requires an API key, ensure it is set via \
                 --api-key or api_key_env in your config.",
                pairs.len(),
                round + 1,
            );
        }

        engine.record_results(&round_results);
        engine.update_current_ratings();

        let need_interim = matches!(strategy, Strategy::TopHeavy) || resolved.live_top.is_some();
        if need_interim && !engine.completed_comparisons.is_empty() {
            let interim = run_scoring(
                &item_ids,
                &engine.completed_comparisons,
                &ScoringOptions {
                    iterations: 200,
                    burn_in: if interim_warm_start.is_some() { 0 } else { 100 },
                    confidence_level: resolved.confidence_level,
                    top_k,
                    warm_start: interim_warm_start.take(),
                    regularization_strength: resolved.regularization_strength,
                    prior_tau2: resolved.prior_tau2,
                    sigma2: resolved.sigma2,
                    proposal_std: resolved.proposal_std,
                    bias_prior_tau2: resolved.bias_prior_tau2,
                    bias_proposal_std: resolved.bias_proposal_std,
                    bias_prior_logit: resolved.bias_prior_logit,
                    decisiveness_prior_tau2: resolved.decisiveness_prior_tau2,
                    decisiveness_proposal_std: resolved.decisiveness_proposal_std,
                },
                &judge_info,
            );
            if matches!(strategy, Strategy::TopHeavy) {
                engine.mcmc_top_k_probs = interim.top_k_probs;
                engine.mcmc_sample_means = interim.sample_means;
            }
            if let Some(limit) = resolved.live_top {
                output::print_live_table(
                    &interim.rankings,
                    &titles,
                    round + 1,
                    total_comparisons,
                    limit,
                );
            }
            interim_warm_start = Some(interim.warm_start_state);
        }
    }

    if cancelled.load(Ordering::Relaxed) {
        eprintln!("\nCancelled. {} comparisons completed before interrupt.", total_comparisons);
    }

    if total_comparisons == 0 {
        bail("All comparisons failed. No results to score.");
    }

    if resolved.verbose {
        eprintln!("Running final MCMC scoring ({total_comparisons} comparisons)...");
    }

    // Final scoring with full MCMC
    let scoring_result = run_scoring(
        &item_ids,
        &engine.completed_comparisons,
        &ScoringOptions {
            iterations: resolved.mcmc_iterations,
            burn_in: resolved.mcmc_burn_in,
            confidence_level: resolved.confidence_level,
            top_k: 0,
            warm_start: None,
            regularization_strength: resolved.regularization_strength,
            prior_tau2: resolved.prior_tau2,
            sigma2: resolved.sigma2,
            proposal_std: resolved.proposal_std,
            bias_prior_tau2: resolved.bias_prior_tau2,
            bias_proposal_std: resolved.bias_proposal_std,
            bias_prior_logit: resolved.bias_prior_logit,
            decisiveness_prior_tau2: resolved.decisiveness_prior_tau2,
            decisiveness_proposal_std: resolved.decisiveness_proposal_std,
        },
        &judge_info,
    );

    if resolved.verbose {
        if total_retries > 0 {
            eprintln!("HTTP retries: {total_retries}");
        }
        if failed_http > 0 {
            eprintln!("HTTP failures (after exhausting retries): {failed_http}");
        }
        if failed_parse > 0 {
            eprintln!("Unparseable responses: {failed_parse}");
        }
    }

    // Print max_tokens warnings (always, not just verbose)
    let mut any_max_tokens_hit = false;
    for (i, judge) in judges.iter().enumerate() {
        if judge_stats[i].max_tokens_hits > 0 {
            any_max_tokens_hit = true;
            eprintln!(
                "Warning: {} hit max_tokens on {}/{} responses.",
                judge.display_name, judge_stats[i].max_tokens_hits, judge_stats[i].total_responses,
            );
        }
    }
    if any_max_tokens_hit {
        eprintln!("Consider increasing max_tokens or adjusting the length instruction in the prompt.");
    }

    // Build judge_id → display_name and token count maps for output
    let judge_names: HashMap<u64, String> = judges.iter()
        .map(|j| (j.judge_id, j.display_name.clone()))
        .collect();
    let judge_tokens: HashMap<u64, (u64, u64)> = judges.iter().enumerate()
        .map(|(i, j)| (j.judge_id, (judge_stats[i].input_tokens, judge_stats[i].output_tokens)))
        .collect();
    let judge_avg_wall_time: HashMap<u64, f64> = judges.iter().enumerate()
        .map(|(i, j)| {
            let avg = if judge_stats[i].round_count > 0 {
                judge_stats[i].wall_time_sum / judge_stats[i].round_count as f64
            } else {
                0.0
            };
            (j.judge_id, avg)
        })
        .collect();

    if resolved.json {
        output::print_json(&scoring_result.rankings, &titles, rounds, total_comparisons, &scoring_result.judge_analytics);
    } else {
        output::print_table(
            &scoring_result.rankings,
            &titles,
            &engine.games_played,
            rounds,
            total_comparisons,
            resolved.confidence_level,
            &scoring_result.judge_analytics,
            &judge_names,
            &judge_tokens,
            &judge_avg_wall_time,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_criteria_single() {
        let criteria = parse_criteria("Which fruit is healthier?");
        assert_eq!(criteria, vec!["Which fruit is healthier?"]);
    }

    #[test]
    fn test_parse_criteria_multiple() {
        let input = "Which fruit is healthier?\n---CRITERION---\nWhich of these two fruits are healthiest?\n---CRITERION---\nRegarding health, which fruit should I opt for?";
        let criteria = parse_criteria(input);
        assert_eq!(criteria, vec![
            "Which fruit is healthier?",
            "Which of these two fruits are healthiest?",
            "Regarding health, which fruit should I opt for?",
        ]);
    }

    #[test]
    fn test_parse_criteria_trims_whitespace() {
        let input = "  first criterion  \n---CRITERION---\n\n  second criterion  \n";
        let criteria = parse_criteria(input);
        assert_eq!(criteria, vec!["first criterion", "second criterion"]);
    }

    #[test]
    fn test_parse_criteria_skips_empty_sections() {
        let input = "---CRITERION---\nonly this one\n---CRITERION---\n---CRITERION---";
        let criteria = parse_criteria(input);
        assert_eq!(criteria, vec!["only this one"]);
    }

    #[test]
    fn test_parse_criteria_empty_input() {
        let criteria = parse_criteria("");
        assert!(criteria.is_empty());
    }

    #[test]
    fn test_cumulative_balancing_equal_weights() {
        let weights = vec![1.0 / 3.0; 3];
        let mut cumulative = vec![0usize; 3];
        let mut rng = rand::rng();

        // Simulate 5 rounds of 10 pairs each
        let mut total = 0;
        for _ in 0..5 {
            total += 10;
            assign_pairs_to_judges(10, &weights, &mut cumulative, total, &mut rng);
        }

        // After 50 total pairs with equal weights, each should have ~16-17
        assert_eq!(cumulative.iter().sum::<usize>(), 50);
        for &count in &cumulative {
            assert!(count >= 16 && count <= 17, "count {count} not in [16, 17]");
        }
    }

    #[test]
    fn test_cumulative_balancing_uneven_round_sizes() {
        let weights = vec![0.5, 0.5];
        let mut cumulative = vec![0usize; 2];
        let mut rng = rand::rng();

        // Odd-sized rounds: 3, 3, 3
        let mut total = 0;
        for _ in 0..3 {
            total += 3;
            assign_pairs_to_judges(3, &weights, &mut cumulative, total, &mut rng);
        }

        // 9 pairs split across 2 judges: should be 4 and 5
        assert_eq!(cumulative.iter().sum::<usize>(), 9);
        assert!(cumulative[0] >= 4 && cumulative[0] <= 5);
        assert!(cumulative[1] >= 4 && cumulative[1] <= 5);
    }
}
