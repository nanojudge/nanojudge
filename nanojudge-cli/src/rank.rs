use nanojudge_core::{
    ComparisonInput, EngineConfig, JudgeInfo, RankingEngine, ScoringOptions, ComparisonDistribution,
    calculate_pairs_for_round, calculate_rounds_for_target_comparisons,
    calculate_total_expected_comparisons, calculate_triples_for_round, run_scoring,
    winner_dist_to_edges,
};
use nanojudge_core::seed;
use rand::seq::SliceRandom;
use reqwest::Client;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::args::{OutputFormat, RankArgs};
use crate::bail;
use crate::config;
use crate::items::load_items;
use crate::llm::{LlmConfig, compare_pair, compare_triple};
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

/// Temper a parsed verdict distribution before it becomes edges:
/// q_i ← q_i^(1/temperature), renormalized — equivalent to dividing each
/// derived edge's log-odds by `temperature`. Values > 1 pull overconfident
/// verdicts toward uniform; 1.0 is the identity. One-hot (text-mode) verdicts
/// are fixed points, so only logprob-derived verdicts are affected.
fn temper_verdict<const N: usize>(mut dist: [f64; N], temperature: f64) -> [f64; N] {
    if temperature == 1.0 {
        return dist;
    }
    let inv = 1.0 / temperature;
    let mut sum = 0.0;
    for q in dist.iter_mut() {
        *q = q.powf(inv);
        sum += *q;
    }
    for q in dist.iter_mut() {
        *q /= sum;
    }
    dist
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

    // Three-item comparisons run a separate acquisition loop; the pairwise path
    // below is left untouched.
    if resolved.items_per_comparison == 3 {
        run_three_way(&args, &config_path, &cfg, &resolved).await;
        return;
    }

    // Resolve judges from [[judge]] blocks
    let judges = resolve_judges(&args.cfg, &cfg, &config_path, resolved.reasoning_enabled);
    let logprobs_mode = judges[0].logprobs;

    if !logprobs_mode {
        eprintln!("Warning: Running without logprobs. Requires more comparisons to reach equivalent accuracy as when using logprobs.");
    }

    let (titles, texts) = load_items(&args);
    let item_ids: Vec<i64> = (0..texts.len() as i64).collect();

    // The anchor rank must exist: fail here, before any LLM spend, rather than
    // at the first scoring pass. Only top-heavy uses the anchor.
    if matches!(resolved.comparison_distribution, ComparisonDistribution::TopHeavy)
        && resolved.anchor_index > (texts.len().saturating_sub(1)) as f64
    {
        bail(format!(
            "anchor-index={} exceeds the last rank ({}) for {} items",
            resolved.anchor_index,
            texts.len().saturating_sub(1),
            texts.len(),
        ));
    }

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
            chat_template_kwargs: j.chat_template_kwargs.clone(),
        })
    }).collect();

    let judge_semaphores: Vec<Arc<tokio::sync::Semaphore>> = judges.iter()
        .map(|j| Arc::new(tokio::sync::Semaphore::new(j.concurrency)))
        .collect();

    // Compute normalized weights for pair assignment
    let total_weight: f64 = judges.iter().map(|j| j.weight).sum();
    let normalized_weights: Vec<f64> = judges.iter().map(|j| j.weight / total_weight).collect();
    // Per-judge min_logprob_coverage values
    let judge_min_logprob_coverages: Vec<f64> = judges.iter().map(|j| j.min_logprob_coverage).collect();
    let judge_verdict_temperatures: Vec<f64> = judges.iter().map(|j| j.verdict_temperature).collect();

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

    if prompt_template.contains("$name1") || prompt_template.contains("$name2") {
        let mut seen = HashMap::new();
        for (i, title) in titles.iter().enumerate() {
            if let Some(prev) = seen.insert(title.as_str(), i) {
                bail(format!(
                    "Template uses $name1/$name2 but items {} and {} have the same name {:?}. \
                     Rename one so the LLM can distinguish them in verdict lines.",
                    prev + 1, i + 1, title,
                ));
            }
        }
    }

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
            eprintln!(
                "Endpoint: {} | Model: {} | Verdict temperature: {}",
                judges[0].endpoint, judges[0].model, judges[0].verdict_temperature,
            );
        } else {
            eprintln!("Judge panel ({} judges):", judges.len());
            for j in &judges {
                eprintln!(
                    "  {} — {} (concurrency: {}, weight: {:.0}%, verdict temp: {})",
                    j.display_name,
                    j.endpoint,
                    j.concurrency,
                    j.weight / total_weight * 100.0,
                    j.verdict_temperature,
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

    let comparison_distribution = resolved.comparison_distribution;

    // Top-heavy selection: when active, each interim scoring pass turns the
    // posterior into per-item pairing weights (sharpened uncertainty ratio
    // around the anchor).
    // `None` for uniform, which needs no selection weights.
    let selection_sharpness = match comparison_distribution {
        ComparisonDistribution::TopHeavy => Some(resolved.selection_sharpness),
        ComparisonDistribution::Uniform => None,
    };

    let engine_config = EngineConfig {
        comparison_distribution,
        matchmaking_sharpness: resolved.matchmaking_sharpness,
        min_uniform_games: resolved.min_uniform_games,
        seed: resolved.seed,
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
    let mut judge_assign_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_JUDGE_ASSIGN);
    let mut jitter_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_TEMP_JITTER);

    let mut interim_warm_start: Option<nanojudge_core::WarmStartState> = None;
    // When --emit-round-rankings is set, record (cumulative comparisons, ranked
    // names) after each round so the benchmark can plot per-round convergence.
    let mut round_rankings: Vec<(usize, Vec<String>)> = Vec::new();
    let pairs_per_round = calculate_pairs_for_round(texts.len());
    // Set when --stop-confidence is active and an interim fit shows the
    // partition confidence — P(every item on its side of the anchor) — has
    // reached the requested level; ends the round loop early and proceeds
    // straight to final scoring.
    let mut early_stop = false;
    // Rounds actually executed (early stop or cancellation can end the run
    // before the configured budget); this is what the output reports.
    let mut rounds_run = 0usize;
    for round in 0..rounds {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        rounds_run = round + 1;

        // Decide how many refit chunks this round splits into. The engine
        // applies the policy: only top-heavy batches may be subdivided (the
        // uniform stage pairs every item exactly once per full round, so it
        // stays whole). Each chunk runs its own scoring refit and re-derives
        // selection weights, so pairing adapts mid-round.
        let chunk_sizes = engine.round_chunk_sizes(resolved.refits_per_round);

        if resolved.verbose {
            eprintln!(
                "Round {}/{}: {} pairs in {} refit chunk(s)",
                round + 1, rounds, pairs_per_round, chunk_sizes.len(),
            );
        }

        for chunk_size in chunk_sizes {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let pairs = engine.generate_pairs(chunk_size);
        let round_start = std::time::Instant::now();

        cumulative_total_pairs += pairs.len();
        let pair_assignments = assign_pairs_to_judges(
            pairs.len(),
            &normalized_weights,
            &mut cumulative_judge_pairs,
            cumulative_total_pairs,
            &mut judge_assign_rng,
        );

        let criterion_weights: Vec<f64> = vec![1.0 / criteria.len() as f64; criteria.len()];
        let criterion_assignments = assign_pairs_to_judges(
            pairs.len(),
            &criterion_weights,
            &mut cumulative_criterion_pairs,
            cumulative_total_pairs,
            &mut judge_assign_rng,
        );

        let mut handles = Vec::with_capacity(pairs.len());

        let precomputed_temperatures: Vec<f64> = pairs.iter().enumerate().map(|(pair_idx, _)| {
            let judge_idx = pair_assignments[pair_idx];
            let base = judge_llm_configs[judge_idx].temperature;
            let jitter = judge_llm_configs[judge_idx].temperature_jitter;
            crate::llm::jittered_temperature(base, jitter, &mut jitter_rng)
        }).collect();

        for (pair_idx, (id_a, id_b)) in pairs.iter().enumerate() {
            let judge_idx = pair_assignments[pair_idx];
            let sem = judge_semaphores[judge_idx].clone();
            let client = client.clone();
            let base_config = &judge_llm_configs[judge_idx];
            let llm_config = Arc::new(LlmConfig {
                temperature: precomputed_temperatures[pair_idx],
                temperature_jitter: 0.0,
                endpoint: base_config.endpoint.clone(),
                model: base_config.model.clone(),
                api_key: base_config.api_key.clone(),
                presence_penalty: base_config.presence_penalty,
                top_p: base_config.top_p,
                logprobs: base_config.logprobs,
                max_tokens: base_config.max_tokens,
                reasoning_effort: base_config.reasoning_effort.clone(),
                chat_template_kwargs: base_config.chat_template_kwargs.clone(),
            });
            let texts = texts.clone();
            let titles = titles.clone();
            let criterion = criteria[criterion_assignments[pair_idx]].clone();
            let analysis_length = analysis_length.clone();
            let template = prompt_template.clone();
            let id_a = *id_a;
            let id_b = *id_b;
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
                    &titles[id_a as usize],
                    &titles[id_b as usize],
                    id_a,
                    id_b,
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
                    if let Some(category_probs) = result.parse_result.category_probs {
                        if let Some(ref file_mutex) = save_file {
                            let line = serde_json::json!({
                                "round": round + 1,
                                "item1": titles[result.item1_id as usize],
                                "item2": titles[result.item2_id as usize],
                                "category_probs": category_probs,
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "response": result.response_text,
                            });
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }

                        // The raw distribution is what went to the JSONL above;
                        // the tempered one is what the scoring engine sees.
                        round_results.push(ComparisonInput::pairwise(
                            result.item1_id,
                            result.item2_id,
                            temper_verdict(category_probs, judge_verdict_temperatures[judge_idx]),
                            assigned_judge_id,
                        ));
                    } else {
                        failed_parse += 1;
                        if let Some(ref file_mutex) = failures_file {
                            let line = serde_json::json!({
                                "round": round + 1,
                                "item1": titles[result.item1_id as usize],
                                "item2": titles[result.item2_id as usize],
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "prompt": result.prompt,
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

        let need_interim = matches!(comparison_distribution, ComparisonDistribution::TopHeavy)
            || resolved.live_top.is_some()
            || resolved.emit_round_rankings;
        if need_interim && !engine.completed_comparisons.is_empty() {
            let interim = run_scoring(
                &item_ids,
                &engine.completed_comparisons,
                &ScoringOptions {
                    iterations: resolved.mcmc_iterations,
                    burn_in: if interim_warm_start.is_some() { 0 } else { 100 },
                    confidence_level: resolved.confidence_level,
                    selection_sharpness,
                    anchor_index: resolved.anchor_index,
                    selection_cutoff: resolved.selection_cutoff,
                    selection_coverage: resolved.selection_coverage,
                    target_prior_games: resolved.target_prior_games,
                    warm_start: interim_warm_start.take(),
                    regularization_strength: resolved.regularization_strength,
                    prior_tau2: resolved.prior_tau2,
                    proposal_std: resolved.proposal_std,
                    bias_prior_tau2: resolved.bias_prior_tau2,
                    bias_proposal_std: resolved.bias_proposal_std,
                    bias_prior_logit: resolved.bias_prior_logit,
                    seed: resolved.seed,
                    inference: resolved.inference,
                },
                &judge_info,
            );
            if resolved.emit_round_rankings {
                let order: Vec<String> = interim
                    .rankings
                    .iter()
                    .map(|r| titles[r.item as usize].clone())
                    .collect();
                round_rankings.push((total_comparisons, order));
            }
            if matches!(comparison_distribution, ComparisonDistribution::TopHeavy) {
                // The interim posterior replaces the per-chunk MLE refit as the
                // matchmaking rating source; its stds let opponent selection
                // integrate over rating uncertainty.
                engine.set_current_posterior(&interim.item_means, &interim.item_stds);
                if let Some(c) = resolved.stop_confidence {
                    // Stop once P(every checked item is on its side of the
                    // anchor) >= c — the log-domain partition confidence
                    // clearing ln(c).
                    let log_conf = interim.partition_log_confidence
                        .expect("top-heavy interim scoring always computes selection ratios");
                    if log_conf >= c.ln() {
                        eprintln!(
                            "Early stop: P(every item on its side of the anchor) = {:.1}% >= {:.1}% after {} comparisons.",
                            log_conf.exp() * 100.0, c * 100.0, total_comparisons
                        );
                        early_stop = true;
                    }
                }
                engine.selection_weights = interim.selection_weights;
            } else {
                // Interim fit ran only for display (--live-top or round
                // rankings on a uniform run): keep MLE ratings so a display
                // flag never changes pairing.
                engine.update_current_ratings();
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
        } else {
            engine.update_current_ratings();
        }
        if early_stop {
            break;
        }
        }
        if early_stop {
            break;
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
            selection_sharpness: None,
            anchor_index: resolved.anchor_index,
            selection_cutoff: resolved.selection_cutoff,
            selection_coverage: resolved.selection_coverage,
            target_prior_games: resolved.target_prior_games,
            warm_start: None,
            regularization_strength: resolved.regularization_strength,
            prior_tau2: resolved.prior_tau2,
            proposal_std: resolved.proposal_std,
            bias_prior_tau2: resolved.bias_prior_tau2,
            bias_proposal_std: resolved.bias_proposal_std,
            bias_prior_logit: resolved.bias_prior_logit,
            seed: resolved.seed,
            inference: resolved.inference,
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
    // Suppressed when reasoning is disabled — max_tokens is intentionally low.
    if resolved.reasoning_enabled {
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

    match resolved.output_format {
        OutputFormat::Json => output::print_json(
            &scoring_result.rankings,
            &titles,
            &engine.games_played,
            rounds_run,
            total_comparisons,
            &scoring_result.judge_analytics,
            scoring_result.panel_positional_bias,
            scoring_result.panel_positional_bias_ci,
            if resolved.emit_round_rankings {
                Some(round_rankings.as_slice())
            } else {
                None
            },
        ),
        OutputFormat::Table => output::print_table(
            &scoring_result.rankings,
            &titles,
            &engine.games_played,
            rounds_run,
            total_comparisons,
            resolved.confidence_level,
            &scoring_result.judge_analytics,
            &judge_names,
            &judge_tokens,
            &judge_avg_wall_time,
        ),
    }
}

/// Three-way (3-item) acquisition loop. Selects triples, asks each judge to rank
/// them, folds the winner-distribution into up to three pairwise edges, and feeds
/// those to the same scoring engine the pairwise path uses. `total_comparisons`
/// here counts LLM calls (triples), so accuracy-per-call is directly comparable
/// to pairwise; each call contributes up to three edges to the fit.
async fn run_three_way(
    args: &RankArgs,
    config_path: &Path,
    cfg: &config::NanojudgeConfig,
    resolved: &crate::resolve::ResolvedConfig,
) {
    let mut judges = resolve_judges(&args.cfg, cfg, config_path, resolved.reasoning_enabled);
    let logprobs_mode = judges[0].logprobs;

    // No-reasoning mode forces max_tokens to 16 (calibrated for a single verdict
    // line). The 3-way ranking output is three lines (~20 tokens), which 16
    // truncates — cutting off the third line so the parser discards it. Give it
    // enough room.
    if !resolved.reasoning_enabled {
        for j in &mut judges {
            j.max_tokens = 48;
        }
    }

    if !logprobs_mode {
        eprintln!("Warning: three-way without logprobs keeps only each triple's winner (2 edges instead of 3, dropping the 2nd-vs-3rd comparison). Enable logprobs for full information.");
    }

    let (titles, texts) = load_items(args);
    let item_ids: Vec<i64> = (0..texts.len() as i64).collect();

    if texts.len() < 3 {
        bail(format!("Three-way comparisons need at least 3 items; got {}.", texts.len()));
    }

    if matches!(resolved.comparison_distribution, ComparisonDistribution::TopHeavy)
        && resolved.anchor_index > (texts.len().saturating_sub(1)) as f64
    {
        bail(format!(
            "anchor-index={} exceeds the last rank ({}) for {} items",
            resolved.anchor_index,
            texts.len().saturating_sub(1),
            texts.len(),
        ));
    }

    let triples_per_round = calculate_triples_for_round(texts.len());

    let rounds = if let Some(target) = resolved.comparisons {
        let r = target.div_ceil(triples_per_round);
        if r == 0 {
            bail(format!(
                "--comparisons ({target}) is less than one round ({triples_per_round} three-way comparisons). Use at least {triples_per_round}.",
            ));
        }
        let actual = triples_per_round * r;
        if actual != target {
            eprintln!(
                "Running {} three-way comparisons instead of {}, to align with round boundaries ({} per round).",
                actual, target, triples_per_round,
            );
        }
        r
    } else {
        resolved.rounds.unwrap_or_else(|| {
            bail(format!("No rounds or comparisons specified. Pass --rounds, --comparisons, or set it in {}", config_path.display()));
        })
    };

    let judge_ids: Vec<u64> = judges.iter().map(|j| j.judge_id).collect();
    let judge_info = JudgeInfo {
        judge_ids: judge_ids.clone(),
        logprobs_mode,
    };

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
            chat_template_kwargs: j.chat_template_kwargs.clone(),
        })
    }).collect();

    let judge_semaphores: Vec<Arc<tokio::sync::Semaphore>> = judges.iter()
        .map(|j| Arc::new(tokio::sync::Semaphore::new(j.concurrency)))
        .collect();

    let total_weight: f64 = judges.iter().map(|j| j.weight).sum();
    let normalized_weights: Vec<f64> = judges.iter().map(|j| j.weight / total_weight).collect();
    let judge_min_logprob_coverages: Vec<f64> = judges.iter().map(|j| j.min_logprob_coverage).collect();
    let judge_verdict_temperatures: Vec<f64> = judges.iter().map(|j| j.verdict_temperature).collect();

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

    let template = Arc::new(resolved.prompt_template.clone());

    let client = Client::new();
    let titles = Arc::new(titles);
    let texts = Arc::new(texts);

    let total_planned = triples_per_round * rounds;
    if resolved.verbose {
        eprintln!(
            "Ranking {} items across {} rounds ({} three-way comparisons planned)",
            texts.len(), rounds, total_planned,
        );
        if criteria.len() == 1 {
            eprintln!("Criterion: \"{}\"", criteria[0]);
        }
        if judges.len() == 1 {
            eprintln!(
                "Endpoint: {} | Model: {} | Verdict temperature: {}",
                judges[0].endpoint, judges[0].model, judges[0].verdict_temperature,
            );
        }
    }

    let save_file = if let Some(ref save_path) = resolved.save_comparisons {
        let path = resolve_save_path(save_path, "comparisons");
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));
        if resolved.verbose {
            eprintln!("Saving comparisons to {}", path.display());
        }
        Some(std::sync::Mutex::new(file))
    } else {
        None
    };

    let failures_file = if let Some(ref save_path) = resolved.save_failures {
        let path = resolve_save_path(save_path, "failures");
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));
        Some(std::sync::Mutex::new(file))
    } else {
        None
    };

    let comparison_distribution = resolved.comparison_distribution;
    let selection_sharpness = match comparison_distribution {
        ComparisonDistribution::TopHeavy => Some(resolved.selection_sharpness),
        ComparisonDistribution::Uniform => None,
    };

    let engine_config = EngineConfig {
        comparison_distribution,
        matchmaking_sharpness: resolved.matchmaking_sharpness,
        min_uniform_games: resolved.min_uniform_games,
        seed: resolved.seed,
    };
    let mut engine = RankingEngine::new(&item_ids, engine_config);

    let analysis_length = resolved.analysis_length.clone();
    let max_retries = resolved.retries;

    let judge_display_names: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.display_name.clone()).collect());
    let judge_models: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.model.clone()).collect());
    let judge_endpoints: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.endpoint.clone()).collect());

    // total_comparisons counts LLM calls (successfully-parsed triples).
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
    let mut judge_assign_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_JUDGE_ASSIGN);
    let mut jitter_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_TEMP_JITTER);
    // RNG for shuffling the three items into random presentation slots (A/B/C).
    // Combined with the engine's per-slot bias correction, this both keeps slot
    // placement unbiased and lets the bias be estimated out.
    let mut slot_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_EDGE_ORIENTATION);

    let mut interim_warm_start: Option<nanojudge_core::WarmStartState> = None;
    let mut round_rankings: Vec<(usize, Vec<String>)> = Vec::new();

    // Set when --stop-confidence is active and an interim fit shows the
    // partition confidence — P(every item on its side of the anchor) — has
    // reached the requested level; ends the round loop early and proceeds
    // straight to final scoring.
    let mut early_stop = false;
    // Rounds actually executed (early stop or cancellation can end the run
    // before the configured budget); this is what the output reports.
    let mut rounds_run = 0usize;
    for round in 0..rounds {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        rounds_run = round + 1;

        // Split the round into refit chunks (only top-heavy subdivides); each
        // chunk runs its own scoring refit and re-derives selection weights.
        let chunk_sizes = engine.round_chunk_sizes_triples(resolved.refits_per_round);
        if resolved.verbose {
            eprintln!(
                "Round {}/{}: {} triples in {} refit chunk(s)",
                round + 1, rounds, triples_per_round, chunk_sizes.len(),
            );
        }

        for chunk_size in chunk_sizes {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let triples = engine.generate_triples(chunk_size);
        let round_start = std::time::Instant::now();

        cumulative_total_pairs += triples.len();
        let judge_assignments = assign_pairs_to_judges(
            triples.len(), &normalized_weights, &mut cumulative_judge_pairs,
            cumulative_total_pairs, &mut judge_assign_rng,
        );
        let criterion_weights: Vec<f64> = vec![1.0 / criteria.len() as f64; criteria.len()];
        let criterion_assignments = assign_pairs_to_judges(
            triples.len(), &criterion_weights, &mut cumulative_criterion_pairs,
            cumulative_total_pairs, &mut judge_assign_rng,
        );

        let precomputed_temperatures: Vec<f64> = (0..triples.len()).map(|idx| {
            let judge_idx = judge_assignments[idx];
            let base = judge_llm_configs[judge_idx].temperature;
            let jitter = judge_llm_configs[judge_idx].temperature_jitter;
            crate::llm::jittered_temperature(base, jitter, &mut jitter_rng)
        }).collect();

        let mut handles = Vec::with_capacity(triples.len());
        for (triple_idx, &(t_a, t_b, t_c)) in triples.iter().enumerate() {
            // Shuffle the three items into random slot order so no item has a
            // fixed presentation position (each of the 6 permutations equally
            // likely). winner_dist stays aligned because compare_triple maps
            // option A/B/C to whatever ids we pass here.
            let mut slots = [t_a, t_b, t_c];
            slots.shuffle(&mut slot_rng);
            let (id_a, id_b, id_c) = (slots[0], slots[1], slots[2]);
            let judge_idx = judge_assignments[triple_idx];
            let sem = judge_semaphores[judge_idx].clone();
            let client = client.clone();
            let base_config = &judge_llm_configs[judge_idx];
            let llm_config = Arc::new(LlmConfig {
                temperature: precomputed_temperatures[triple_idx],
                temperature_jitter: 0.0,
                endpoint: base_config.endpoint.clone(),
                model: base_config.model.clone(),
                api_key: base_config.api_key.clone(),
                presence_penalty: base_config.presence_penalty,
                top_p: base_config.top_p,
                logprobs: base_config.logprobs,
                max_tokens: base_config.max_tokens,
                reasoning_effort: base_config.reasoning_effort.clone(),
                chat_template_kwargs: base_config.chat_template_kwargs.clone(),
            });
            let texts = texts.clone();
            let criterion = criteria[criterion_assignments[triple_idx]].clone();
            let analysis_length = analysis_length.clone();
            let template = template.clone();
            let min_logprob_coverage = judge_min_logprob_coverages[judge_idx];
            let assigned_judge_id = judge_ids[judge_idx];
            let judge_name = judge_display_names[judge_idx].clone();
            let verbose = resolved.verbose;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let result = compare_triple(
                    &client, &llm_config, &template, &criterion,
                    &texts[id_a as usize], &texts[id_b as usize], &texts[id_c as usize],
                    id_a, id_b, id_c,
                    min_logprob_coverage, &analysis_length, max_retries, verbose, &judge_name,
                ).await;
                (result, assigned_judge_id, judge_idx, std::time::Instant::now())
            });
            handles.push((handle, judge_idx));
        }

        let mut round_results: Vec<ComparisonInput> = Vec::new();
        let mut round_llm_calls: usize = 0;
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
                Ok((Ok(tw), assigned_judge_id, judge_idx, finished_at)) => {
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    total_retries += tw.retries_used;
                    judge_stats[judge_idx].total_responses += 1;
                    if tw.hit_max_tokens {
                        judge_stats[judge_idx].max_tokens_hits += 1;
                    }
                    if let Some(usage) = &tw.usage {
                        judge_stats[judge_idx].input_tokens += usage.prompt_tokens;
                        judge_stats[judge_idx].output_tokens += usage.completion_tokens;
                    }
                    if let Some(winner_dist) = tw.winner_dist {
                        round_llm_calls += 1;
                        if let Some(ref file_mutex) = save_file {
                            let line = serde_json::json!({
                                "round": round + 1,
                                "item1": titles[tw.item_a_id as usize],
                                "item2": titles[tw.item_b_id as usize],
                                "item3": titles[tw.item_c_id as usize],
                                "winner_dist": winner_dist,
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "response": tw.response_text,
                            });
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }

                        // Each edge already carries the true presentation slots
                        // (from the shuffled A/B/C order), so the scoring engine
                        // estimates and corrects the per-slot positional bias — no
                        // orientation coin needed. Feed the edges as-is.
                        let edges = winner_dist_to_edges(
                            [tw.item_a_id, tw.item_b_id, tw.item_c_id],
                            // Raw distribution went to the JSONL above; the
                            // scoring engine sees the tempered one.
                            temper_verdict(winner_dist, judge_verdict_temperatures[judge_idx]),
                            assigned_judge_id,
                        );
                        round_results.extend(edges);
                    } else {
                        failed_parse += 1;
                        if let Some(ref file_mutex) = failures_file {
                            let line = serde_json::json!({
                                "round": round + 1,
                                "item1": titles[tw.item_a_id as usize],
                                "item2": titles[tw.item_b_id as usize],
                                "item3": titles[tw.item_c_id as usize],
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "prompt": tw.prompt,
                                "response": tw.response_text,
                            });
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }
                        if resolved.verbose {
                            eprintln!(
                                "  Warning: unparseable 3-way ranking for {} / {} / {}, skipping",
                                titles[tw.item_a_id as usize],
                                titles[tw.item_b_id as usize],
                                titles[tw.item_c_id as usize],
                            );
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
            for (i, judge) in judges.iter().enumerate() {
                if judge_aborted[i] > 0 {
                    eprintln!("  {} had {} in-flight requests", judge.display_name, judge_aborted[i]);
                }
            }
            break;
        }

        for (j, finish) in judge_last_finish.iter().enumerate() {
            if let Some(t) = finish {
                judge_stats[j].wall_time_sum += t.duration_since(round_start).as_secs_f64();
                judge_stats[j].round_count += 1;
            }
        }

        total_comparisons += round_llm_calls;

        if resolved.verbose {
            eprintln!(
                "  Completed: {} successful, {} failed ({} edges fed)",
                round_llm_calls, triples.len() - round_llm_calls, round_results.len(),
            );
        }

        if round_llm_calls == 0 && !triples.is_empty() {
            eprintln!(
                "Warning: all {} three-way comparisons in a round-{} chunk failed.",
                triples.len(), round + 1,
            );
        }

        engine.record_results(&round_results);

        let need_interim = matches!(comparison_distribution, ComparisonDistribution::TopHeavy)
            || resolved.live_top.is_some()
            || resolved.emit_round_rankings;
        if need_interim && !engine.completed_comparisons.is_empty() {
            let interim = run_scoring(
                &item_ids,
                &engine.completed_comparisons,
                &ScoringOptions {
                    iterations: resolved.mcmc_iterations,
                    burn_in: if interim_warm_start.is_some() { 0 } else { 100 },
                    confidence_level: resolved.confidence_level,
                    selection_sharpness,
                    anchor_index: resolved.anchor_index,
                    selection_cutoff: resolved.selection_cutoff,
                    selection_coverage: resolved.selection_coverage,
                    target_prior_games: resolved.target_prior_games,
                    warm_start: interim_warm_start.take(),
                    regularization_strength: resolved.regularization_strength,
                    prior_tau2: resolved.prior_tau2,
                    proposal_std: resolved.proposal_std,
                    bias_prior_tau2: resolved.bias_prior_tau2,
                    bias_proposal_std: resolved.bias_proposal_std,
                    bias_prior_logit: resolved.bias_prior_logit,
                    seed: resolved.seed,
                    inference: resolved.inference,
                },
                &judge_info,
            );
            if resolved.emit_round_rankings {
                let order: Vec<String> = interim.rankings.iter()
                    .map(|r| titles[r.item as usize].clone())
                    .collect();
                round_rankings.push((total_comparisons, order));
            }
            if matches!(comparison_distribution, ComparisonDistribution::TopHeavy) {
                // The interim posterior replaces the per-chunk MLE refit as the
                // matchmaking rating source; its stds let opponent selection
                // integrate over rating uncertainty.
                engine.set_current_posterior(&interim.item_means, &interim.item_stds);
                if let Some(c) = resolved.stop_confidence {
                    // Stop once P(every checked item is on its side of the
                    // anchor) >= c — the log-domain partition confidence
                    // clearing ln(c).
                    let log_conf = interim.partition_log_confidence
                        .expect("top-heavy interim scoring always computes selection ratios");
                    if log_conf >= c.ln() {
                        eprintln!(
                            "Early stop: P(every item on its side of the anchor) = {:.1}% >= {:.1}% after {} comparisons.",
                            log_conf.exp() * 100.0, c * 100.0, total_comparisons
                        );
                        early_stop = true;
                    }
                }
                engine.selection_weights = interim.selection_weights;
            } else {
                // Interim fit ran only for display (--live-top or round
                // rankings on a uniform run): keep MLE ratings so a display
                // flag never changes pairing.
                engine.update_current_ratings();
            }
            if let Some(limit) = resolved.live_top {
                output::print_live_table(&interim.rankings, &titles, round + 1, total_comparisons, limit);
            }
            interim_warm_start = Some(interim.warm_start_state);
        } else {
            engine.update_current_ratings();
        }
        if early_stop {
            break;
        }
        }
        if early_stop {
            break;
        }
    }

    if cancelled.load(Ordering::Relaxed) {
        eprintln!("\nCancelled. {} three-way comparisons completed before interrupt.", total_comparisons);
    }

    if total_comparisons == 0 {
        bail("All three-way comparisons failed. No results to score.");
    }

    if resolved.verbose {
        eprintln!("Running final MCMC scoring ({} three-way comparisons, {} edges)...",
            total_comparisons, engine.completed_comparisons.len());
    }

    let scoring_result = run_scoring(
        &item_ids,
        &engine.completed_comparisons,
        &ScoringOptions {
            iterations: resolved.mcmc_iterations,
            burn_in: resolved.mcmc_burn_in,
            confidence_level: resolved.confidence_level,
            selection_sharpness: None,
            anchor_index: resolved.anchor_index,
            selection_cutoff: resolved.selection_cutoff,
            selection_coverage: resolved.selection_coverage,
            target_prior_games: resolved.target_prior_games,
            warm_start: None,
            regularization_strength: resolved.regularization_strength,
            prior_tau2: resolved.prior_tau2,
            proposal_std: resolved.proposal_std,
            bias_prior_tau2: resolved.bias_prior_tau2,
            bias_proposal_std: resolved.bias_proposal_std,
            bias_prior_logit: resolved.bias_prior_logit,
            seed: resolved.seed,
            inference: resolved.inference,
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
            eprintln!("Unparseable rankings: {failed_parse}");
        }
    }

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

    match resolved.output_format {
        OutputFormat::Json => output::print_json(
            &scoring_result.rankings, &titles, &engine.games_played, rounds_run, total_comparisons,
            &scoring_result.judge_analytics,
            scoring_result.panel_positional_bias, scoring_result.panel_positional_bias_ci,
            if resolved.emit_round_rankings { Some(round_rankings.as_slice()) } else { None },
        ),
        OutputFormat::Table => output::print_table(
            &scoring_result.rankings, &titles, &engine.games_played, rounds_run, total_comparisons,
            resolved.confidence_level, &scoring_result.judge_analytics,
            &judge_names, &judge_tokens, &judge_avg_wall_time,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temper_verdict_identity_at_one() {
        assert_eq!(temper_verdict([0.9, 0.1], 1.0), [0.9, 0.1]);
        assert_eq!(temper_verdict([0.7, 0.2, 0.1], 1.0), [0.7, 0.2, 0.1]);
    }

    #[test]
    fn test_temper_verdict_one_hot_fixed_point() {
        // Text-mode verdicts are one-hot: 1^(1/T)=1 and 0^(1/T)=0, so
        // tempering must leave them exactly alone.
        assert_eq!(temper_verdict([1.0, 0.0], 4.0), [1.0, 0.0]);
        assert_eq!(temper_verdict([0.0, 1.0], 4.0), [0.0, 1.0]);
        assert_eq!(temper_verdict([0.0, 1.0, 0.0], 4.0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_temper_verdict_divides_log_odds() {
        let t = temper_verdict([0.999, 0.001], 4.0);
        assert!((t[0] + t[1] - 1.0).abs() < 1e-12);
        let expected_log_odds = (0.999f64 / 0.001).ln() / 4.0;
        assert!(((t[0] / t[1]).ln() - expected_log_odds).abs() < 1e-9);
        // Overconfident 0.999 gets pulled well toward 0.5
        assert!(t[0] < 0.9 && t[0] > 0.5);
    }

    #[test]
    fn test_temper_verdict_sharpens_below_one() {
        let t = temper_verdict([0.7, 0.3], 0.5);
        let expected_log_odds = (0.7f64 / 0.3).ln() / 0.5;
        assert!(((t[0] / t[1]).ln() - expected_log_odds).abs() < 1e-9);
        assert!(t[0] > 0.7);
    }

    #[test]
    fn test_temper_verdict_three_way_pairwise_ratios() {
        // Tempering the 3-vector before the Luce ratio must divide every
        // pairwise edge's log-odds by T.
        let q = [0.9, 0.08, 0.02];
        let t = temper_verdict(q, 4.0);
        assert!((t.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let expected = (q[i] / q[j]).ln() / 4.0;
            assert!(((t[i] / t[j]).ln() - expected).abs() < 1e-9);
        }
    }

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
            assert!((16..=17).contains(&count), "count {count} not in [16, 17]");
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
