use nanojudge_core::{
    Edge, EngineConfig, JudgeInfo, RankingEngine, ScoringOptions,
    JudgementDistribution, calculate_budget,
    judgements_needed_for_every_item_to_appear_once, run_scoring, item_hash, judge_hash, prompt_hash,
    lineup_verdict_to_edges, LineupVerdict,
};
use nanojudge_core::seed;
use rand::seq::SliceRandom;
use reqwest::Client;
use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::OnceCell;

use crate::args::{OutputFormat, RankArgs};
use crate::bail;
use crate::config;
use crate::items::load_items;
use crate::llm::{LineupJudgementResult, LlmConfig, PairJudgementResult, judge_pair, judge_lineup};
use crate::output;
use crate::parse::ParseResult;
use crate::prompt::{build_lineup_prompt, build_prompt};
use crate::resolve::{resolve_config, resolve_judgements_per_item, resolve_judges, ResolvedConfig, ResolvedJudge};

#[derive(Default)]
struct JudgeStats {
    input_tokens: u64,
    output_tokens: u64,
    max_tokens_hits: usize,
    total_responses: usize,
    wall_time_sum: f64,
    collection_count: usize,
}

/// Print max_tokens warnings (always, not just verbose).
fn print_max_tokens_warnings(judges: &[ResolvedJudge], judge_stats: &[JudgeStats]) {
    let mut any_max_tokens_hit = false;
    for (judge, stats) in judges.iter().zip(judge_stats) {
        if stats.max_tokens_hits > 0 {
            any_max_tokens_hit = true;
            eprintln!(
                "Warning: {} hit max_tokens on {}/{} responses.",
                judge.display_name, stats.max_tokens_hits, stats.total_responses,
            );
        }
    }
    if any_max_tokens_hit {
        eprintln!("Consider increasing max_tokens or adjusting the length instruction in the prompt.");
    }
}

/// One judgement and the temperature its request was sent at.
struct Judged<R> {
    result: R,
    temperature: f64,
}

/// Judgement results keyed by (judge index, prompt hash). A judge with
/// deliberation off and `reasoning_effort = "none"` answers an identical prompt
/// identically when its verdict isn't sampled: in logprobs mode, or in text
/// mode at temperature 0. A repeat then reuses the stored result instead of
/// being sent again. It still counts as a judgement of its own, like any other.
/// Loaded judgements fill the cache too (see `loaded_reuse_entries`).
type ReuseCache<R> = HashMap<(usize, u64), ReuseCell<R>>;

/// One prompt's entry in a `ReuseCache`: empty until a judgement with a verdict
/// fills it.
type ReuseCell<R> = Arc<OnceCell<Arc<Judged<R>>>>;

/// Why a judgement did not fill its reuse cell.
enum Unfilled<R> {
    Failed(String),
    NoVerdict(Arc<Judged<R>>),
}

/// Send a judgement, or reuse the identical prompt's result when `cell` is
/// given. Concurrent copies of a prompt wait for the first to finish. Only a
/// judgement with a verdict fills the cell, so after a failed request or an
/// unparseable reply the next copy is sent for real. Returns the judgement and
/// whether it was reused.
async fn judge_or_reuse<R, Fut>(
    cell: Option<ReuseCell<R>>,
    has_verdict: fn(&R) -> bool,
    send: impl FnOnce() -> Fut,
) -> Result<(Arc<Judged<R>>, bool), String>
where
    Fut: Future<Output = Result<Judged<R>, String>>,
{
    let Some(cell) = cell else {
        return send().await.map(|judged| (Arc::new(judged), false));
    };
    let mut sent = false;
    let outcome = cell
        .get_or_try_init(|| async {
            sent = true;
            let judged = Arc::new(send().await.map_err(Unfilled::Failed)?);
            if has_verdict(&judged.result) {
                Ok(judged)
            } else {
                Err(Unfilled::NoVerdict(judged))
            }
        })
        .await;
    match outcome {
        Ok(judged) => Ok((judged.clone(), !sent)),
        Err(Unfilled::Failed(e)) => Err(e),
        Err(Unfilled::NoVerdict(judged)) => Ok((judged, false)),
    }
}

/// Whether a judge's repeated prompts reuse earlier results (see `ReuseCache`).
fn reuses_results(judge: &ResolvedJudge, deliberation_enabled: bool) -> bool {
    !deliberation_enabled
        && judge.reasoning_effort.as_deref() == Some("none")
        && (judge.logprobs || judge.temperature == 0.0)
}

/// What one judgement asks: its items in presentation order and the prompt.
struct Slot {
    item_ids: Vec<i64>,
    prompt: Arc<String>,
    prompt_hash: u64,
}

impl Slot {
    fn new(item_ids: Vec<i64>, prompt: String) -> Self {
        let prompt_hash = prompt_hash(&prompt);
        Slot { item_ids, prompt: Arc::new(prompt), prompt_hash }
    }
}

/// A loaded judgement that repeats of its prompt in this run reuse. `record`
/// passed `load_edges`, so its verdict fields are valid.
struct LoadedReuse {
    key: (usize, u64),
    record: serde_json::Value,
    temperature: f64,
    hit_max_tokens: bool,
    response_text: Option<String>,
    reasoning_text: Option<String>,
}

/// Pick the loaded judgements this run can reuse. A loaded judgement qualifies
/// when its judge reuses results in this run and it was made with the same
/// deliberation, `reasoning_effort`, `provider` and `chat_template_kwargs`,
/// and, in text mode, at temperature 0. Every judge in `records` must be in
/// `judges`.
fn loaded_reuse_entries(
    records: Vec<serde_json::Value>,
    judges: &[ResolvedJudge],
    deliberation_enabled: bool,
    path: &Path,
) -> Vec<LoadedReuse> {
    let field = |record: &serde_json::Value, name: &str| -> serde_json::Value {
        match record.get(name) {
            Some(v) => v.clone(),
            None => bail(format!("A judgement in {} has no {name} field", path.display())),
        }
    };
    let malformed = |name: &str| -> ! {
        bail(format!("A judgement in {} has a malformed {name} field", path.display()))
    };
    let mut entries = Vec::new();
    for record in records {
        let judge_id = judge_hash(
            record["judge_endpoint"].as_str().expect("load_edges requires judge_endpoint"),
            record["judge_model"].as_str().expect("load_edges requires judge_model"),
        );
        let judge_idx = judges
            .iter()
            .position(|j| j.judge_id == judge_id)
            .expect("seed_prior_edges requires every loaded judge to be in the run");
        let judge = &judges[judge_idx];
        if !reuses_results(judge, deliberation_enabled) {
            continue;
        }
        let same_settings = field(&record, "deliberation") == serde_json::json!(deliberation_enabled)
            && field(&record, "reasoning_effort") == serde_json::json!(judge.reasoning_effort)
            && field(&record, "provider") == serde_json::json!(judge.provider)
            && field(&record, "chat_template_kwargs") == serde_json::json!(judge.chat_template_kwargs);
        if !same_settings {
            continue;
        }
        let hash = match field(&record, "prompt_hash").as_str() {
            Some(h) if h.len() == 16 => u64::from_str_radix(h, 16).unwrap_or_else(|_| malformed("prompt_hash")),
            _ => malformed("prompt_hash"),
        };
        let temperature = field(&record, "temperature").as_f64().unwrap_or_else(|| malformed("temperature"));
        if !judge.logprobs && temperature != 0.0 {
            continue;
        }
        let hit_max_tokens = field(&record, "hit_max_tokens").as_bool().unwrap_or_else(|| malformed("hit_max_tokens"));
        // A text field is absent when its run didn't save it, and null when the
        // endpoint returned none; either way there is no text to carry over.
        let text = |name: &str| record.get(name).and_then(|v| v.as_str()).map(str::to_owned);
        let response_text = text("response");
        let reasoning_text = text("reasoning");
        entries.push(LoadedReuse {
            key: (judge_idx, hash),
            record,
            temperature,
            hit_max_tokens,
            response_text,
            reasoning_text,
        });
    }
    entries
}

/// Fill a reuse cache from loaded judgements, keeping the first of any repeats.
fn seed_reuse_cache<R>(
    loaded: Vec<LoadedReuse>,
    result: fn(&serde_json::Value, Option<String>, Option<String>, bool) -> R,
) -> ReuseCache<R> {
    let mut cache: ReuseCache<R> = HashMap::new();
    for l in loaded {
        cache.entry(l.key).or_insert_with(|| {
            let judged = Judged {
                result: result(&l.record, l.response_text, l.reasoning_text, l.hit_max_tokens),
                temperature: l.temperature,
            };
            Arc::new(OnceCell::from(Arc::new(judged)))
        });
    }
    cache
}

/// A loaded pair record as a reusable result. It sent no request, so it
/// carries no retries or usage.
fn loaded_pair_result(
    record: &serde_json::Value,
    response_text: Option<String>,
    reasoning_text: Option<String>,
    hit_max_tokens: bool,
) -> PairJudgementResult {
    let probs = &record["category_probs"];
    let prob = |i: usize| probs[i].as_f64().expect("load_edges validates category_probs");
    PairJudgementResult {
        parse_result: ParseResult { category_probs: Some([prob(0), prob(1)]) },
        response_text,
        reasoning_text,
        retries_used: 0,
        usage: None,
        hit_max_tokens,
    }
}

/// A loaded lineup record as a reusable result, like `loaded_pair_result`.
fn loaded_lineup_result(
    record: &serde_json::Value,
    response_text: Option<String>,
    reasoning_text: Option<String>,
    hit_max_tokens: bool,
) -> LineupJudgementResult {
    let numbers = |name: &str| record[name].as_array().expect("load_edges validates lineup verdicts");
    let verdict = LineupVerdict {
        ranking: numbers("ranking").iter().map(|v| v.as_u64().expect("load_edges validates ranking") as usize).collect(),
        place_probs: numbers("place_probs").iter().map(|v| v.as_f64().expect("load_edges validates place_probs")).collect(),
    };
    LineupJudgementResult {
        verdict: Some(verdict),
        response_text,
        reasoning_text,
        retries_used: 0,
        usage: None,
        hit_max_tokens,
    }
}

fn resolve_save_path(path: &Path, prefix: &str) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    if path.is_dir() {
        path.join(format!("{prefix}-{ts}.jsonl"))
    } else {
        // A bare filename has an empty parent, not None; treat it as ".".
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        if !parent.exists() {
            bail(format!("Directory {} does not exist", parent.display()));
        }
        path.to_path_buf()
    }
}

/// Temper a parsed verdict distribution before it becomes an edge:
/// q_i ← q_i^(1/temperature), renormalized — equivalent to dividing the
/// edge's log-odds by `temperature`. Values > 1 pull overconfident verdicts
/// toward uniform; 1.0 is the identity. One-hot (text-mode) verdicts are fixed
/// points, so only logprob-derived verdicts are affected.
pub(crate) fn temper_verdict<const N: usize>(mut dist: [f64; N], temperature: f64) -> [f64; N] {
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

/// Temper every edge decomposed from a lineup verdict. Tempering each edge's
/// `[p, 1 - p]` is the same as tempering the lineup's Luce strengths
/// themselves, since an edge depends only on the ratio of its two strengths.
pub(crate) fn temper_edges_in_place(edges: &mut [Edge], temperature: f64) {
    for e in edges.iter_mut() {
        e.category_probs = temper_verdict(e.category_probs, temperature);
    }
}

fn sort_and_dedup_items(
    titles: Vec<String>,
    texts: Vec<String>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let text_hashes: Vec<String> = texts.iter().map(|t| format!("{:016x}", item_hash(t))).collect();

    let mut order: Vec<usize> = (0..texts.len()).collect();
    order.sort_by(|&a, &b| text_hashes[a].cmp(&text_hashes[b]));
    let titles: Vec<String> = order.iter().map(|&i| titles[i].clone()).collect();
    let texts: Vec<String> = order.iter().map(|&i| texts[i].clone()).collect();
    let text_hashes: Vec<String> = order.iter().map(|&i| text_hashes[i].clone()).collect();

    for i in 1..text_hashes.len() {
        if text_hashes[i] == text_hashes[i - 1] {
            let (a, b) = (order[i - 1] + 1, order[i] + 1);
            bail(format!(
                "Items {} and {} have identical text (hash {}). Remove the duplicate.",
                a.min(b), a.max(b), text_hashes[i],
            ));
        }
    }

    (titles, texts, text_hashes)
}

/// Parse a criterion file into a list of criteria, splitting on ---CRITERION---.
fn parse_criteria(content: &str) -> Vec<String> {
    content
        .split("---CRITERION---")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Assign the judgement attempts scheduled before one refit, balancing cumulative usage.
///
/// `cumulative_total` includes these judgements. Each judge's target is
/// `cumulative_total * weight`, minus what they have already been assigned. This
/// ensures even distribution over time rather than independent per-refit allocation.
/// Updates `cumulative_assigned` in place.
fn assign_judgements_to_judges(
    judgements_before_refit: usize,
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
        let target_this_refit = (w * cumulative_total as f64) - cumulative_assigned[i] as f64;
        let floor = (target_this_refit.floor() as usize)
            .min(judgements_before_refit.saturating_sub(assigned));
        counts.push(floor);
        remainders.push((i, target_this_refit - floor as f64));
        assigned += floor;
    }

    remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(judge_idx, _) in remainders.iter().take(judgements_before_refit - assigned) {
        counts[judge_idx] += 1;
    }

    for (i, &count) in counts.iter().enumerate() {
        cumulative_assigned[i] += count;
    }

    let mut assignments: Vec<usize> = Vec::with_capacity(judgements_before_refit);
    for (judge_idx, &count) in counts.iter().enumerate() {
        assignments.extend(std::iter::repeat_n(judge_idx, count));
    }
    assignments.shuffle(rng);

    assignments
}

/// Load a previously saved judgements file and seed its edges into `engine`
/// before any new judgements are collected. Enforces the load-time invariants
/// (matching lineup size — via `load_edges` — plus matching logprobs mode and
/// every judge present in the current panel), remaps loaded items onto this
/// run's indices by text hash (skipping any edge whose items are not in the
/// current run), and records the survivors so they inform pairing and scoring.
/// Loaded edges are kept out of the judge balancer's ledger by design: this
/// only ever touches the engine. Returns how many loaded judgements contributed
/// at least one edge to this run, and their records.
fn seed_prior_edges(
    engine: &mut RankingEngine,
    path: &Path,
    text_hashes: &[String],
    lineup_size: usize,
    run_judge_ids: &[u64],
    run_logprobs_mode: bool,
    per_judge_verdict_temperatures: &HashMap<String, f64>,
    verbose: bool,
) -> (usize, Vec<serde_json::Value>) {
    let (mut edges, edge_judgements, _names, file_judge_ids, file_judge_display_names, _flag_keys,
         total_judgements, file_logprobs_mode, _temps, hash_keys, records) =
        crate::score::load_edges(path, None, per_judge_verdict_temperatures, Some(lineup_size));

    if edges.is_empty() {
        bail(format!("No valid judgements found in {}", path.display()));
    }

    // Mixing one-hot verdicts (text mode) and full distributions (logprobs mode)
    // under one global flag is not well-defined, so require an exact match.
    if file_logprobs_mode != run_logprobs_mode {
        bail(format!(
            "{} was saved in {} mode, but this run uses {} mode. Loading requires a matching logprobs mode.",
            path.display(),
            if file_logprobs_mode { "logprobs" } else { "text" },
            if run_logprobs_mode { "logprobs" } else { "text" },
        ));
    }

    // Every judge that produced a loaded edge must be a configured judge here, so
    // its verdict temperature is known and its position bias is estimated during
    // scoring. To reuse a judge's data without new comparisons, configure it with
    // weight 0 rather than dropping it.
    for (i, jid) in file_judge_ids.iter().enumerate() {
        if !run_judge_ids.contains(jid) {
            bail(format!(
                "Loaded file {} contains judge \"{}\", which is not in this run's judge panel. \
                 Every judge in a loaded file must be a configured judge here. Add it to your \
                 panel with weight 0 if you don't want new comparisons from it.",
                path.display(), file_judge_display_names[i],
            ));
        }
    }

    // Remap loaded item indices onto this run's items by text hash. `hash_keys`
    // is aligned with the loaded edges' item indices; `text_hashes` is this run's
    // hash-sorted item list, so its position is the engine item id.
    let hash_to_idx: HashMap<&str, i64> = text_hashes
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i as i64))
        .collect();

    let mut seeded: Vec<Edge> = Vec::with_capacity(edges.len());
    let mut covered: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut contributing: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut skipped = 0usize;
    for (edge, &judgement) in edges.iter_mut().zip(&edge_judgements) {
        let h1 = hash_keys[edge.item1 as usize].as_str();
        let h2 = hash_keys[edge.item2 as usize].as_str();
        match (hash_to_idx.get(h1), hash_to_idx.get(h2)) {
            (Some(&i1), Some(&i2)) => {
                edge.item1 = i1;
                edge.item2 = i2;
                covered.insert(i1);
                covered.insert(i2);
                contributing.insert(judgement);
                seeded.push(*edge);
            }
            _ => skipped += 1,
        }
    }

    if seeded.is_empty() {
        bail(format!(
            "None of the {} loaded judgements in {} reference items in this run. \
             Check that the file and the item list belong together.",
            total_judgements, path.display(),
        ));
    }

    engine.record_edges(&seeded);

    if verbose {
        eprintln!(
            "Loaded {} prior judgements from {} ({} edges seeded, covering {} items, {} judges)",
            total_judgements, path.display(), seeded.len(), covered.len(), file_judge_ids.len(),
        );
    }
    // Always report a partial loss: silently discarding part of a loaded file
    // would leave a degraded run indistinguishable from a clean one without -v.
    if skipped > 0 {
        eprintln!("{skipped} of {} loaded edges skipped (referenced items not in this run)", edges.len());
    }
    let contributing_records = records
        .into_iter()
        .enumerate()
        .filter(|(i, _)| contributing.contains(i))
        .map(|(_, record)| record)
        .collect();
    (contributing.len(), contributing_records)
}

/// Everything both runs build before their refit loop: judges, items, budget,
/// per-judge request config, criteria, output files, the engine (with any
/// loaded judgements seeded) and the Ctrl-C handler.
struct RunSetup {
    judges: Vec<ResolvedJudge>,
    logprobs_mode: bool,
    titles: Arc<Vec<String>>,
    texts: Arc<Vec<String>>,
    text_hashes: Vec<String>,
    item_ids: Vec<i64>,
    budget: usize,
    judgements_per_refit: usize,
    judge_ids: Vec<u64>,
    judge_info: JudgeInfo,
    judge_llm_configs: Vec<Arc<LlmConfig>>,
    judge_semaphores: Vec<Arc<tokio::sync::Semaphore>>,
    normalized_weights: Vec<f64>,
    judge_min_logprob_coverages: Vec<f64>,
    judge_verdict_temperatures: Vec<f64>,
    criteria: Vec<String>,
    template: Arc<String>,
    client: Client,
    save_file: Option<std::sync::Mutex<std::fs::File>>,
    failures_file: Option<std::sync::Mutex<std::fs::File>>,
    engine: RankingEngine,
    /// Loaded judgements that contributed edges to this run; None without
    /// `--load-judgements`. Added to displayed judgement counts only.
    loaded_judgements: Option<usize>,
    /// Loaded judgements that repeats of their prompts reuse.
    loaded_reuse: Vec<LoadedReuse>,
    interim_scoring_options: ScoringOptions,
    judge_display_names: Arc<Vec<String>>,
    judge_models: Arc<Vec<String>>,
    judge_endpoints: Arc<Vec<String>>,
    judge_reasoning_efforts: Arc<Vec<Option<String>>>,
    cancelled: Arc<AtomicBool>,
}

/// Build the shared run state. All validation happens here, before any LLM
/// spend. `noun` names one judgement in messages ("judgements" or "lineup
/// judgements").
async fn setup_run(
    args: &RankArgs,
    config_path: &Path,
    cfg: &config::NanojudgeConfig,
    resolved: &ResolvedConfig,
    judgements_per_item: usize,
    noun: &str,
) -> RunSetup {
    // Resolve judges from [[judge]] blocks
    let judges = resolve_judges(&args.cfg, cfg, config_path, resolved.deliberation_enabled, resolved.lineup_size);
    let logprobs_mode = judges[0].logprobs;

    if !logprobs_mode {
        eprintln!("Warning: Running without logprobs. Requires more judgements to reach equivalent accuracy as when using logprobs.");
    }

    let (titles, texts) = load_items(args);
    let (titles, texts, text_hashes) = sort_and_dedup_items(titles, texts);

    let item_ids: Vec<i64> = (0..texts.len() as i64).collect();

    let lineup_size = resolved.lineup_size;
    if texts.len() < lineup_size {
        bail(format!(
            "lineup-size={} needs at least {} items; got {}.",
            lineup_size, lineup_size, texts.len()
        ));
    }

    // The anchor rank must exist: fail here, before any LLM spend, rather than
    // at the first scoring pass. Only top-heavy uses the anchor.
    if matches!(resolved.judgement_distribution, JudgementDistribution::TopHeavy)
        && resolved.anchor_index > (texts.len().saturating_sub(1)) as f64
    {
        bail(format!(
            "anchor-index={} exceeds the last rank ({}) for {} items",
            resolved.anchor_index,
            texts.len().saturating_sub(1),
            texts.len(),
        ));
    }

    let budget = calculate_budget(texts.len(), judgements_per_item, lineup_size);
    let default_judgements_per_refit =
        judgements_needed_for_every_item_to_appear_once(texts.len(), lineup_size);
    let judgements_per_refit = resolved
        .judgements_per_refit
        .unwrap_or(default_judgements_per_refit);

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
            provider: j.provider.clone(),
            request_timeout: j.request_timeout,
        })
    }).collect();

    let judge_semaphores: Vec<Arc<tokio::sync::Semaphore>> = judges.iter()
        .map(|j| Arc::new(tokio::sync::Semaphore::new(j.concurrency)))
        .collect();

    // Compute normalized weights for judgement assignment
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

    let template = Arc::new(resolved.prompt_template.clone());

    if template.contains("$name1") || template.contains("$name2") {
        let mut seen = HashMap::new();
        for (i, title) in titles.iter().enumerate() {
            if let Some(_prev) = seen.insert(title.as_str(), i) {
                bail(format!(
                    "Template uses $name1/$name2 but multiple items have the same name {:?}. \
                     Rename one so the LLM can distinguish them in verdict lines.",
                    title,
                ));
            }
        }
    }

    // Weight-0 judges only contribute prior data; they send no requests.
    if !args.skip_probe {
        crate::probe::check_judges(judges.iter().filter(|j| j.weight > 0.0)).await;
    }

    let client = Client::new();
    let titles = Arc::new(titles);
    let texts = Arc::new(texts);

    if resolved.verbose {
        eprintln!(
            "Ranking {} items ({} {noun} planned, {} per item)",
            texts.len(),
            budget,
            judgements_per_item,
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

    // Set up judgement saving if requested
    let save_file = if let Some(ref save_path) = resolved.save_successful_judgements {
        let path = resolve_save_path(save_path, "judgements");

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));

        if resolved.verbose {
            eprintln!("Saving successful judgements to {}", path.display());
        }

        Some(std::sync::Mutex::new(file))
    } else {
        None
    };

    // Set up failure saving if requested
    let failures_file = if let Some(ref save_path) = resolved.save_failed_judgements {
        let path = resolve_save_path(save_path, "failures");

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));

        if resolved.verbose {
            eprintln!("Saving failed judgements to {}", path.display());
        }

        Some(std::sync::Mutex::new(file))
    } else {
        None
    };

    let judgement_distribution = resolved.judgement_distribution;

    // Top-heavy selection: when active, each interim scoring pass turns the
    // posterior into per-item selection weights (sharpened uncertainty ratio
    // around the anchor).
    // `None` for uniform, which needs no selection weights.
    let selection_sharpness = match judgement_distribution {
        JudgementDistribution::TopHeavy => Some(resolved.selection_sharpness),
        JudgementDistribution::Uniform => None,
    };

    let engine_config = EngineConfig {
        judgement_distribution,
        matchmaking_sharpness: resolved.matchmaking_sharpness,
        min_uniform_edges: resolved.min_uniform_edges,
        regularization_strength: resolved.regularization_strength,
        seed: resolved.seed,
    };
    let mut engine = RankingEngine::new(&item_ids, engine_config);

    // Seed prior judgements (if any) before the collection loop, so they inform
    // the very first selection pass and the final ranking. Kept out of the judge
    // balancer's ledger — this only touches the engine. The lineup-size guard in
    // load_edges rejects any file whose judgements are not this run's lineup size.
    let (loaded_judgements, loaded_reuse) = if let Some(ref load_path) = args.load_judgements {
        let per_judge_verdict_temps: HashMap<String, f64> = judges
            .iter()
            .map(|j| (format!("{}@{}", j.model, j.endpoint), j.verdict_temperature))
            .collect();
        let (loaded, records) = seed_prior_edges(
            &mut engine,
            load_path,
            &text_hashes,
            lineup_size,
            &judge_ids,
            logprobs_mode,
            &per_judge_verdict_temps,
            resolved.verbose,
        );
        let reuse = loaded_reuse_entries(records, &judges, resolved.deliberation_enabled, load_path);
        (Some(loaded), reuse)
    } else {
        (None, Vec::new())
    };

    // Scoring options are constant across refits, so build them once and share
    // them between the pre-loop priming pass and the per-refit interim scoring.
    let interim_scoring_options = ScoringOptions {
        confidence_level: resolved.confidence_level,
        selection_sharpness,
        anchor_index: resolved.anchor_index,
        selection_cutoff: resolved.selection_cutoff,
        selection_coverage: resolved.selection_coverage,
        target_prior_edges: resolved.target_prior_edges,
        regularization_strength: resolved.regularization_strength,
        prior_tau2: resolved.prior_tau2,
        bias_prior_tau2: resolved.bias_prior_tau2,
        bias_prior_logit: resolved.bias_prior_logit,
    };

    // Judge display names (Arc for sharing across tasks)
    let judge_display_names: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.display_name.clone()).collect());
    let judge_models: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.model.clone()).collect());
    let judge_endpoints: Arc<Vec<String>> = Arc::new(judges.iter().map(|j| j.endpoint.clone()).collect());
    let judge_reasoning_efforts: Arc<Vec<Option<String>>> = Arc::new(judges.iter().map(|j| j.reasoning_effort.clone()).collect());

    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let cancelled = cancelled.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\nCancelling... (press Ctrl-C again to force quit)");
            cancelled.store(true, Ordering::Relaxed);
            let _ = tokio::signal::ctrl_c().await;
            std::process::exit(130);
        });
    }

    RunSetup {
        judges,
        logprobs_mode,
        titles,
        texts,
        text_hashes,
        item_ids,
        budget,
        judgements_per_refit,
        judge_ids,
        judge_info,
        judge_llm_configs,
        judge_semaphores,
        normalized_weights,
        judge_min_logprob_coverages,
        judge_verdict_temperatures,
        criteria,
        template,
        client,
        save_file,
        failures_file,
        engine,
        loaded_judgements,
        loaded_reuse,
        interim_scoring_options,
        judge_display_names,
        judge_models,
        judge_endpoints,
        judge_reasoning_efforts,
        cancelled,
    }
}

/// Prime the rating state from any seeded edges before the loop. Without this,
/// the first selection pass runs matchmaking on flat initial ratings, ignoring
/// the loaded data entirely on refit 1. Returns true when the loaded
/// judgements already meet `stop_confidence`.
fn prime_from_seed(
    engine: &mut RankingEngine,
    item_ids: &[i64],
    scoring_options: &ScoringOptions,
    judge_info: &JudgeInfo,
    resolved: &ResolvedConfig,
) -> bool {
    if engine.completed_edges.is_empty() {
        return false;
    }
    let mut early_stop = false;
    if matches!(resolved.judgement_distribution, JudgementDistribution::TopHeavy) {
        // Seeding can already push every item past `min_uniform_edges`, so the
        // first selection pass would be top-heavy and need `selection_weights` —
        // normally produced by the first refit's interim scoring, which runs
        // only after the first `generate_pairs`/`generate_lineups`. Compute them
        // now. And if `stop_confidence` is set and the seed already meets it,
        // stop before collecting anything: the loaded data is already
        // conclusive, so the run scores it and prints the ranking with no new
        // comparisons (this is the deliberate zero-new case, not a failure).
        let primed = run_scoring(
            item_ids,
            &engine.completed_edges,
            scoring_options,
            judge_info,
        );
        engine.set_current_posterior(&primed.item_means, &primed.item_stds);
        if let Some(c) = resolved.stop_confidence {
            let log_conf = primed.partition_log_confidence
                .expect("top-heavy interim scoring always computes selection ratios");
            if log_conf >= c.ln() {
                eprintln!(
                    "Early stop before collecting: loaded judgements already meet stop_confidence \
                     (P(every item on its side of the anchor) = {:.1}% >= {:.1}%). \
                     Collecting no new comparisons.",
                    log_conf.exp() * 100.0, c * 100.0,
                );
                early_stop = true;
            }
        }
        engine.selection_weights = primed.selection_weights;
    } else {
        // Uniform mode uses `current_ratings` for info-gain matchmaking but
        // needs no `selection_weights`. A cheap MLE fit over the seed is enough
        // to make the first pass exploit the loaded data.
        engine.update_current_ratings();
    }
    early_stop
}

/// Update the rating state after a refit's edges are recorded. Runs full
/// interim scoring when something needs it (top-heavy selection weights and
/// early stopping, `--live-top`, `--emit-interim-rankings`); otherwise a cheap
/// MLE fit for matchmaking. Returns true on an early stop. `total_judgements`
/// counts this run's judgements; displayed counts add `loaded_judgements`.
#[allow(clippy::too_many_arguments)]
fn refit_interim_scoring(
    engine: &mut RankingEngine,
    item_ids: &[i64],
    scoring_options: &ScoringOptions,
    judge_info: &JudgeInfo,
    resolved: &ResolvedConfig,
    titles: &[String],
    interim_rankings: &mut Vec<(usize, Vec<String>)>,
    refits_run: usize,
    total_judgements: usize,
    loaded_judgements: Option<usize>,
) -> bool {
    let top_heavy = matches!(resolved.judgement_distribution, JudgementDistribution::TopHeavy);
    let need_interim = top_heavy
        || resolved.live_top.is_some()
        || resolved.emit_interim_rankings;
    if !need_interim || engine.completed_edges.is_empty() {
        engine.update_current_ratings();
        return false;
    }

    let shown_judgements = total_judgements + loaded_judgements.unwrap_or(0);
    let mut early_stop = false;
    let interim = run_scoring(
        item_ids,
        &engine.completed_edges,
        scoring_options,
        judge_info,
    );
    if resolved.emit_interim_rankings {
        let ranked_names: Vec<String> = interim
            .rankings
            .iter()
            .map(|r| titles[r.item as usize].clone())
            .collect();
        interim_rankings.push((shown_judgements, ranked_names));
    }
    if top_heavy {
        engine.set_current_posterior(&interim.item_means, &interim.item_stds);
        if let Some(c) = resolved.stop_confidence {
            let log_conf = interim.partition_log_confidence
                .expect("top-heavy interim scoring always computes selection ratios");
            if log_conf >= c.ln() {
                eprintln!(
                    "Early stop: P(every item on its side of the anchor) = {:.1}% >= {:.1}% after {}.",
                    log_conf.exp() * 100.0, c * 100.0,
                    output::judgement_count(shown_judgements, loaded_judgements, "judgements"),
                );
                early_stop = true;
            }
        }
        engine.selection_weights = interim.selection_weights;
    } else {
        engine.update_current_ratings();
    }
    if let Some(limit) = resolved.live_top {
        output::print_live_table(
            &interim.rankings,
            titles,
            refits_run,
            shown_judgements,
            loaded_judgements,
            limit,
        );
    }
    early_stop
}

/// Counters a refit loop accumulates, reported once it ends.
struct RunTotals {
    total_judgements: usize,
    /// From `RunSetup::loaded_judgements`; added to displayed counts only.
    loaded_judgements: Option<usize>,
    judgements_done: usize,
    /// Judgements that reused an identical earlier prompt's result (see
    /// `ReuseCache`). Included in `total_judgements`.
    reused_judgements: usize,
    total_retries: usize,
    failed_http: usize,
    failed_parse: usize,
    judge_stats: Vec<JudgeStats>,
}

/// Bail if nothing usable was collected, run the final scoring, print the
/// failure summary and print the ranking. `noun` as in `setup_run`.
#[allow(clippy::too_many_arguments)]
fn finish_run(
    resolved: &ResolvedConfig,
    noun: &str,
    judges: &[ResolvedJudge],
    judge_info: &JudgeInfo,
    item_ids: &[i64],
    titles: &[String],
    text_hashes: &[String],
    engine: &RankingEngine,
    totals: RunTotals,
    early_stop: bool,
    cancelled: bool,
    interim_rankings: &[(usize, Vec<String>)],
) {
    let RunTotals {
        total_judgements,
        loaded_judgements,
        judgements_done,
        reused_judgements,
        total_retries,
        failed_http,
        failed_parse,
        judge_stats,
    } = totals;

    if cancelled {
        eprintln!("\nCancelled. {total_judgements} {noun} completed before interrupt.");
    }

    // Zero new judgements is a failure only when we did not deliberately stop
    // (early stop or Ctrl-C). A deliberate stop scores whatever edges exist,
    // including ones seeded from --load-judgements.
    if total_judgements == 0 && !early_stop && !cancelled {
        bail(format!("All {noun} failed."));
    }
    if engine.completed_edges.is_empty() {
        bail(format!("Cancelled before any {noun} completed. No results to score."));
    }

    // Displayed counts include loaded judgements; the bail above and the
    // success ratio below describe this run's requests only.
    let shown_judgements = total_judgements + loaded_judgements.unwrap_or(0);
    if resolved.verbose {
        eprintln!(
            "Running final scoring ({})...",
            output::judgement_count(shown_judgements, loaded_judgements, noun),
        );
    }

    // Final scoring over all completed judgements.
    let scoring_result = run_scoring(
        item_ids,
        &engine.completed_edges,
        &ScoringOptions {
            confidence_level: resolved.confidence_level,
            selection_sharpness: None,
            anchor_index: resolved.anchor_index,
            selection_cutoff: resolved.selection_cutoff,
            selection_coverage: resolved.selection_coverage,
            target_prior_edges: resolved.target_prior_edges,
            regularization_strength: resolved.regularization_strength,
            prior_tau2: resolved.prior_tau2,
            bias_prior_tau2: resolved.bias_prior_tau2,
            bias_prior_logit: resolved.bias_prior_logit,
        },
        judge_info,
    );

    // Failure summary: always printed, not verbose-gated. A degraded run
    // must be distinguishable from a clean one without extra flags.
    if judgements_done > 0 {
        eprintln!(
            "{} of {} judgements succeeded",
            total_judgements, judgements_done
        );
    }
    if reused_judgements > 0 {
        eprintln!("Reused results of identical earlier prompts: {reused_judgements}");
    }
    if total_retries > 0 {
        eprintln!("HTTP retries: {total_retries}");
    }
    if failed_http > 0 {
        eprintln!("HTTP failures: {failed_http}");
    }
    if failed_parse > 0 {
        eprintln!("Unparseable responses: {failed_parse}");
    }

    print_max_tokens_warnings(judges, &judge_stats);

    // Build judge_id → display_name and token count maps for output
    let judge_names: HashMap<u64, String> = judges.iter()
        .map(|j| (j.judge_id, j.display_name.clone()))
        .collect();
    let judge_tokens: HashMap<u64, (u64, u64)> = judges.iter().enumerate()
        .map(|(i, j)| (j.judge_id, (judge_stats[i].input_tokens, judge_stats[i].output_tokens)))
        .collect();
    let judge_avg_wall_time: HashMap<u64, f64> = judges.iter().enumerate()
        .map(|(i, j)| {
            let avg = if judge_stats[i].collection_count > 0 {
                judge_stats[i].wall_time_sum / judge_stats[i].collection_count as f64
            } else {
                0.0
            };
            (j.judge_id, avg)
        })
        .collect();

    match resolved.output_format {
        OutputFormat::Json => output::print_json(
            &scoring_result.rankings,
            titles,
            text_hashes,
            &engine.edge_counts,
            shown_judgements,
            loaded_judgements,
            &scoring_result.judge_analytics,
            scoring_result.panel_positional_bias,
            scoring_result.panel_positional_bias_ci,
            if resolved.emit_interim_rankings {
                Some(interim_rankings)
            } else {
                None
            },
        ),
        OutputFormat::Table => output::print_table(
            &scoring_result.rankings,
            titles,
            &engine.edge_counts,
            shown_judgements,
            loaded_judgements,
            resolved.confidence_level,
            &scoring_result.judge_analytics,
            &judge_names,
            &judge_tokens,
            &judge_avg_wall_time,
            scoring_result.panel_positional_bias,
            scoring_result.panel_positional_bias_ci,
        ),
    }
}

pub async fn run(args: RankArgs) {
    let config_path = args.config.clone().unwrap_or_else(config::config_path);
    let cfg = config::load_config(&config_path);
    let resolved = resolve_config(&args.cfg, &cfg);
    let judgements_per_item = resolve_judgements_per_item(&args.cfg, &cfg);

    // Lineup judgements (3+ items) run their own refit loop; setup and
    // teardown are shared with the pairwise loop below.
    if resolved.lineup_size >= 3 {
        run_lineup_judgements(&args, &config_path, &cfg, &resolved, judgements_per_item).await;
        return;
    }

    let RunSetup {
        judges,
        logprobs_mode,
        titles,
        texts,
        text_hashes,
        item_ids,
        budget,
        judgements_per_refit,
        judge_ids,
        judge_info,
        judge_llm_configs,
        judge_semaphores,
        normalized_weights,
        judge_min_logprob_coverages,
        judge_verdict_temperatures,
        criteria,
        template,
        client,
        save_file,
        failures_file,
        mut engine,
        loaded_judgements,
        loaded_reuse,
        interim_scoring_options,
        judge_display_names,
        judge_models,
        judge_endpoints,
        judge_reasoning_efforts,
        cancelled,
    } = setup_run(&args, &config_path, &cfg, &resolved, judgements_per_item, "judgements").await;

    let save_prompt_text = resolved.save_prompt_text;
    let save_response_text = resolved.save_response_text;
    let save_reasoning_text = resolved.save_reasoning_text;
    let deliberation_length = resolved.deliberation_length.clone();
    let max_retries = resolved.retries;

    let mut total_judgements: usize = 0;
    let mut total_retries: usize = 0;
    let mut failed_http: usize = 0;
    let mut failed_parse: usize = 0;
    let mut judge_stats: Vec<JudgeStats> = (0..judges.len()).map(|_| JudgeStats::default()).collect();

    let mut cumulative_judge_pairs: Vec<usize> = vec![0; judges.len()];
    let mut cumulative_criterion_pairs: Vec<usize> = vec![0; criteria.len()];
    let mut cumulative_total_pairs: usize = 0;
    let mut judge_assign_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_JUDGE_ASSIGN);
    let mut jitter_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_TEMP_JITTER);

    // When --emit-interim-rankings is set, record (cumulative judgements, ranked
    // names) after each refit so the benchmark can plot convergence by refit.
    let mut interim_rankings: Vec<(usize, Vec<String>)> = Vec::new();
    let mut early_stop = prime_from_seed(
        &mut engine, &item_ids, &interim_scoring_options, &judge_info, &resolved,
    );
    let reusing: Vec<bool> = judges.iter().map(|j| reuses_results(j, resolved.deliberation_enabled)).collect();
    let mut reuse_cache = seed_reuse_cache(loaded_reuse, loaded_pair_result);
    let mut reused_judgements = 0usize;
    let mut refits_run = 0usize;
    let mut judgements_done = 0usize;

    while !early_stop && judgements_done < budget {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        refits_run += 1;

        let judgements_before_refit = judgements_per_refit.min(budget - judgements_done);

        if resolved.verbose {
            eprintln!(
                "Refit {}: collecting {} pairs ({}/{})",
                refits_run, judgements_before_refit, judgements_done, budget,
            );
        }

        let pairs = engine.generate_pairs(judgements_before_refit);
        let collection_start = std::time::Instant::now();

        cumulative_total_pairs += pairs.len();
        let pair_assignments = assign_judgements_to_judges(
            pairs.len(),
            &normalized_weights,
            &mut cumulative_judge_pairs,
            cumulative_total_pairs,
            &mut judge_assign_rng,
        );

        let criterion_weights: Vec<f64> = vec![1.0 / criteria.len() as f64; criteria.len()];
        let criterion_assignments = assign_judgements_to_judges(
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
                provider: base_config.provider.clone(),
                request_timeout: base_config.request_timeout,
            });
            let texts = texts.clone();
            let (a, b) = (*id_a as usize, *id_b as usize);
            let slot = Slot::new(vec![*id_a, *id_b], build_prompt(
                &template,
                &criteria[criterion_assignments[pair_idx]],
                &texts[a],
                &texts[b],
                &titles[a],
                &titles[b],
                &deliberation_length,
            ));
            let prompt = slot.prompt.clone();
            let min_logprob_coverage = judge_min_logprob_coverages[judge_idx];
            let assigned_judge_id = judge_ids[judge_idx];
            let judge_name = judge_display_names[judge_idx].clone();
            let temperature = precomputed_temperatures[pair_idx];
            let cell = reusing[judge_idx].then(|| {
                reuse_cache.entry((judge_idx, slot.prompt_hash)).or_default().clone()
            });

            let verbose = resolved.verbose;
            let handle = tokio::spawn(async move {
                let has_verdict = |r: &PairJudgementResult| r.parse_result.category_probs.is_some();
                let result = judge_or_reuse(cell, has_verdict, move || async move {
                    let _permit = sem.acquire().await.unwrap();
                    let result = judge_pair(
                        &client,
                        &llm_config,
                        &prompt,
                        &texts[a],
                        &texts[b],
                        min_logprob_coverage,
                        max_retries,
                        verbose,
                        &judge_name,
                    )
                    .await?;
                    Ok(Judged { result, temperature })
                })
                .await;
                (result, assigned_judge_id, judge_idx, std::time::Instant::now())
            });

            handles.push((handle, judge_idx, slot));
        }

        let mut refit_results: Vec<Edge> = Vec::new();
        let mut judge_last_finish: Vec<Option<std::time::Instant>> = vec![None; judges.len()];
        let mut judge_aborted: Vec<usize> = vec![0; judges.len()];

        for (pair_result_idx, (handle, handle_judge_idx, slot)) in handles.into_iter().enumerate() {
            // Once cancelled, every uncollected request is aborted, including
            // ones that have already finished. Harvesting finished results
            // would keep only the fast responses from the tail of the batch,
            // which selects on response length and judge speed and can bias
            // the fit. Dropping the whole tail keeps the collected set an
            // unbiased spawn-order prefix.
            if cancelled.load(Ordering::Relaxed) {
                handle.abort();
                judge_aborted[handle_judge_idx] += 1;
                continue;
            }
            let cancelled_ref = &cancelled;
            let abort_handle = handle.abort_handle();
            let result = tokio::select! {
                r = handle => r,
                _ = async { while !cancelled_ref.load(Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } } => {
                    abort_handle.abort();
                    judge_aborted[handle_judge_idx] += 1;
                    continue;
                }
            };
            match result {
                Ok((Ok((judged, reused)), assigned_judge_id, judge_idx, finished_at)) => {
                    let result = &judged.result;
                    let actual_temperature = judged.temperature;
                    let criterion = &criteria[criterion_assignments[pair_result_idx]];
                    let (id1, id2) = (slot.item_ids[0], slot.item_ids[1]);
                    // Track latest finish time per judge before this refit.
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    // A reused judgement sent no request, so it adds no retries,
                    // responses or tokens.
                    if reused {
                        reused_judgements += 1;
                    } else {
                        total_retries += result.retries_used;
                        judge_stats[judge_idx].total_responses += 1;
                        if result.hit_max_tokens {
                            judge_stats[judge_idx].max_tokens_hits += 1;
                        }
                        if let Some(usage) = &result.usage {
                            judge_stats[judge_idx].input_tokens += usage.prompt_tokens;
                            judge_stats[judge_idx].output_tokens += usage.completion_tokens;
                        }
                    }
                    if let Some(category_probs) = result.parse_result.category_probs {
                        if let Some(ref file_mutex) = save_file {
                            let mut line = serde_json::json!({
                                "refit": refits_run,
                                "item1": titles[id1 as usize],
                                "item2": titles[id2 as usize],
                                "item1_text_hash": text_hashes[id1 as usize],
                                "item2_text_hash": text_hashes[id2 as usize],
                                "category_probs": category_probs,
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "temperature": actual_temperature,
                                "deliberation": resolved.deliberation_enabled,
                                "reasoning_effort": judge_reasoning_efforts[judge_idx],
                                "provider": judges[judge_idx].provider,
                                "chat_template_kwargs": judges[judge_idx].chat_template_kwargs,
                                "criterion": criterion,
                                "prompt_hash": format!("{:016x}", slot.prompt_hash),
                                "logprobs": logprobs_mode,
                                "retries_used": if reused { 0 } else { result.retries_used },
                                "hit_max_tokens": result.hit_max_tokens,
                                "reused": reused,
                            });
                            if let (false, Some(usage)) = (reused, &result.usage) {
                                line["usage"] = usage.to_json();
                            }
                            if save_prompt_text {
                                line["prompt"] = serde_json::json!(*slot.prompt);
                            }
                            if save_response_text {
                                line["response"] = serde_json::json!(result.response_text);
                            }
                            if save_reasoning_text {
                                line["reasoning"] = serde_json::json!(result.reasoning_text);
                            }
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }

                        // The raw distribution is what went to the JSONL above;
                        // the tempered one is what the scoring engine sees.
                        refit_results.push(Edge::new(
                            id1,
                            id2,
                            temper_verdict(category_probs, judge_verdict_temperatures[judge_idx]),
                            assigned_judge_id,
                        ));
                    } else {
                        failed_parse += 1;
                        if let Some(ref file_mutex) = failures_file {
                            let mut line = serde_json::json!({
                                "refit": refits_run,
                                "item1": titles[id1 as usize],
                                "item2": titles[id2 as usize],
                                "item1_text_hash": text_hashes[id1 as usize],
                                "item2_text_hash": text_hashes[id2 as usize],
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "temperature": actual_temperature,
                                "deliberation": resolved.deliberation_enabled,
                                "reasoning_effort": judge_reasoning_efforts[judge_idx],
                                "provider": judges[judge_idx].provider,
                                "chat_template_kwargs": judges[judge_idx].chat_template_kwargs,
                                "criterion": criterion,
                                "prompt_hash": format!("{:016x}", slot.prompt_hash),
                                "logprobs": logprobs_mode,
                                "retries_used": result.retries_used,
                                "hit_max_tokens": result.hit_max_tokens,
                                "prompt": *slot.prompt,
                                "response": result.response_text,
                                "reasoning": result.reasoning_text,
                            });
                            if let Some(ref usage) = result.usage {
                                line["usage"] = usage.to_json();
                            }
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }
                        if resolved.verbose {
                            if judges.len() > 1 {
                                eprintln!(
                                    "  Warning: unparseable response for {} vs {} [{}], skipping",
                                    titles[id1 as usize],
                                    titles[id2 as usize],
                                    judge_display_names[judge_idx],
                                );
                            } else {
                                eprintln!(
                                    "  Warning: unparseable response for {} vs {}, skipping",
                                    titles[id1 as usize],
                                    titles[id2 as usize],
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
                            "  Error [{}]: {e}",
                            judge_display_names[judge_idx],
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
            // Keep the judgements collected before the interrupt: they are
            // already in the JSONL, and they form a spawn-order prefix of the
            // batch, so keeping them adds no speed-selected data. Aborted
            // requests were never attempted, so they do not count toward the
            // "X of Y succeeded" denominator. Per-judge wall-time stats are
            // skipped: a partial collection is not comparable to a full one.
            let aborted: usize = judge_aborted.iter().sum();
            total_judgements += refit_results.len();
            judgements_done += pairs.len() - aborted;
            engine.record_edges(&refit_results);
            break;
        }

        for (j, finish) in judge_last_finish.iter().enumerate() {
            if let Some(t) = finish {
                judge_stats[j].wall_time_sum += t.duration_since(collection_start).as_secs_f64();
                judge_stats[j].collection_count += 1;
            }
        }

        total_judgements += refit_results.len();
        judgements_done += pairs.len();

        let failed_before_refit = pairs.len() - refit_results.len();
        if resolved.verbose {
            eprintln!(
                "  Completed: {} successful, {} failed",
                refit_results.len(),
                failed_before_refit,
            );
        }

        if failed_before_refit == pairs.len() {
            eprintln!(
                "Warning: all {} judgements before refit {} failed. \
                 If your endpoint requires an API key, ensure it is set via \
                 --api-key or api_key_env in your config.",
                pairs.len(),
                refits_run,
            );
        }

        engine.record_edges(&refit_results);

        early_stop = refit_interim_scoring(
            &mut engine, &item_ids, &interim_scoring_options, &judge_info, &resolved,
            &titles, &mut interim_rankings, refits_run, total_judgements, loaded_judgements,
        );
        if early_stop {
            break;
        }
    }

    finish_run(
        &resolved, "judgements", &judges, &judge_info, &item_ids, &titles, &text_hashes, &engine,
        RunTotals { total_judgements, loaded_judgements, judgements_done, reused_judgements, total_retries, failed_http, failed_parse, judge_stats },
        early_stop, cancelled.load(Ordering::Relaxed), &interim_rankings,
    );
}

/// Lineup acquisition loop. Selects lineups, asks each judge to rank
/// them, decomposes each ranking into one edge per pair in the lineup,
/// and feeds those to the same scoring engine the pairwise path uses.
/// `total_judgements` here counts LLM calls (lineups), so accuracy-per-call is
/// directly comparable to pairwise; each call contributes up to k(k-1)/2 edges
/// to the fit, for a lineup of k items.
async fn run_lineup_judgements(
    args: &RankArgs,
    config_path: &Path,
    cfg: &config::NanojudgeConfig,
    resolved: &ResolvedConfig,
    judgements_per_item: usize,
) {
    let RunSetup {
        judges,
        logprobs_mode,
        titles,
        texts,
        text_hashes,
        item_ids,
        budget,
        judgements_per_refit,
        judge_ids,
        judge_info,
        judge_llm_configs,
        judge_semaphores,
        normalized_weights,
        judge_min_logprob_coverages,
        judge_verdict_temperatures,
        criteria,
        template,
        client,
        save_file,
        failures_file,
        mut engine,
        loaded_judgements,
        loaded_reuse,
        interim_scoring_options,
        judge_display_names,
        judge_models,
        judge_endpoints,
        judge_reasoning_efforts,
        cancelled,
    } = setup_run(args, config_path, cfg, resolved, judgements_per_item, "lineup judgements").await;

    let lineup_size = resolved.lineup_size;
    let save_prompt_text = resolved.save_prompt_text;
    let save_response_text = resolved.save_response_text;
    let save_reasoning_text = resolved.save_reasoning_text;
    let deliberation_length = resolved.deliberation_length.clone();
    let max_retries = resolved.retries;

    // total_judgements counts LLM calls (successfully-parsed lineups).
    let mut total_judgements: usize = 0;
    let mut total_retries: usize = 0;
    let mut failed_http: usize = 0;
    let mut failed_parse: usize = 0;
    let mut judge_stats: Vec<JudgeStats> = (0..judges.len()).map(|_| JudgeStats::default()).collect();

    let mut cumulative_judge_pairs: Vec<usize> = vec![0; judges.len()];
    let mut cumulative_criterion_pairs: Vec<usize> = vec![0; criteria.len()];
    let mut cumulative_total_pairs: usize = 0;
    let mut judge_assign_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_JUDGE_ASSIGN);
    let mut jitter_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_TEMP_JITTER);
    // RNG for shuffling each lineup into random presentation slots (A/B/C/...).
    // Combined with the engine's per-slot bias correction, this both keeps slot
    // placement unbiased and lets the bias be estimated out.
    let mut slot_rng = seed::make_rng(resolved.seed, seed::SUBSYSTEM_EDGE_ORIENTATION);

    let mut interim_rankings: Vec<(usize, Vec<String>)> = Vec::new();
    let mut early_stop = prime_from_seed(
        &mut engine, &item_ids, &interim_scoring_options, &judge_info, resolved,
    );
    let reusing: Vec<bool> = judges.iter().map(|j| reuses_results(j, resolved.deliberation_enabled)).collect();
    let mut reuse_cache = seed_reuse_cache(loaded_reuse, loaded_lineup_result);
    let mut reused_judgements = 0usize;
    let mut refits_run = 0usize;
    let mut judgements_done = 0usize;

    while !early_stop && judgements_done < budget {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        refits_run += 1;

        let judgements_before_refit = judgements_per_refit.min(budget - judgements_done);

        if resolved.verbose {
            eprintln!(
                "Refit {}: collecting {} lineups ({}/{})",
                refits_run, judgements_before_refit, judgements_done, budget,
            );
        }

        let lineups = engine.generate_lineups(judgements_before_refit, lineup_size);
        let collection_start = std::time::Instant::now();

        cumulative_total_pairs += lineups.len();
        let judge_assignments = assign_judgements_to_judges(
            lineups.len(), &normalized_weights, &mut cumulative_judge_pairs,
            cumulative_total_pairs, &mut judge_assign_rng,
        );
        let criterion_weights: Vec<f64> = vec![1.0 / criteria.len() as f64; criteria.len()];
        let criterion_assignments = assign_judgements_to_judges(
            lineups.len(), &criterion_weights, &mut cumulative_criterion_pairs,
            cumulative_total_pairs, &mut judge_assign_rng,
        );

        let precomputed_temperatures: Vec<f64> = (0..lineups.len()).map(|idx| {
            let judge_idx = judge_assignments[idx];
            let base = judge_llm_configs[judge_idx].temperature;
            let jitter = judge_llm_configs[judge_idx].temperature_jitter;
            crate::llm::jittered_temperature(base, jitter, &mut jitter_rng)
        }).collect();

        let mut handles = Vec::with_capacity(lineups.len());
        for (lineup_idx, lineup) in lineups.iter().enumerate() {
            // Shuffle the lineup into random slot order so no item has a fixed
            // presentation position (every permutation equally likely).
            // The verdict indexes options A/B/C/... in this slot order.
            let mut slot_ids = lineup.clone();
            slot_ids.shuffle(&mut slot_rng);
            let judge_idx = judge_assignments[lineup_idx];
            let sem = judge_semaphores[judge_idx].clone();
            let client = client.clone();
            let base_config = &judge_llm_configs[judge_idx];
            let llm_config = Arc::new(LlmConfig {
                temperature: precomputed_temperatures[lineup_idx],
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
                provider: base_config.provider.clone(),
                request_timeout: base_config.request_timeout,
            });
            let option_texts: Vec<&str> = slot_ids.iter().map(|&id| texts[id as usize].as_str()).collect();
            let slot = Slot::new(slot_ids, build_lineup_prompt(
                &template,
                &criteria[criterion_assignments[lineup_idx]],
                &option_texts,
                &deliberation_length,
            ));
            let prompt = slot.prompt.clone();
            let min_logprob_coverage = judge_min_logprob_coverages[judge_idx];
            let assigned_judge_id = judge_ids[judge_idx];
            let judge_name = judge_display_names[judge_idx].clone();
            let verbose = resolved.verbose;
            let temperature = precomputed_temperatures[lineup_idx];
            let cell = reusing[judge_idx].then(|| {
                reuse_cache.entry((judge_idx, slot.prompt_hash)).or_default().clone()
            });

            let handle = tokio::spawn(async move {
                let has_verdict = |r: &LineupJudgementResult| r.verdict.is_some();
                let result = judge_or_reuse(cell, has_verdict, move || async move {
                    let _permit = sem.acquire().await.unwrap();
                    let result = judge_lineup(
                        &client, &llm_config, &prompt, lineup_size,
                        min_logprob_coverage, max_retries, verbose, &judge_name,
                    ).await?;
                    Ok(Judged { result, temperature })
                })
                .await;
                (result, assigned_judge_id, judge_idx, std::time::Instant::now())
            });
            handles.push((handle, judge_idx, slot));
        }

        let mut refit_results: Vec<Edge> = Vec::new();
        let mut calls_before_refit: usize = 0;
        let mut judge_last_finish: Vec<Option<std::time::Instant>> = vec![None; judges.len()];
        let mut judge_aborted: Vec<usize> = vec![0; judges.len()];

        for (lineup_result_idx, (handle, handle_judge_idx, slot)) in handles.into_iter().enumerate() {
            // Once cancelled, every uncollected request is aborted, including
            // ones that have already finished. Harvesting finished results
            // would keep only the fast responses from the tail of the batch,
            // which selects on response length and judge speed and can bias
            // the fit. Dropping the whole tail keeps the collected set an
            // unbiased spawn-order prefix.
            if cancelled.load(Ordering::Relaxed) {
                handle.abort();
                judge_aborted[handle_judge_idx] += 1;
                continue;
            }
            let cancelled_ref = &cancelled;
            let abort_handle = handle.abort_handle();
            let result = tokio::select! {
                r = handle => r,
                _ = async { while !cancelled_ref.load(Ordering::Relaxed) { tokio::time::sleep(std::time::Duration::from_millis(50)).await; } } => {
                    abort_handle.abort();
                    judge_aborted[handle_judge_idx] += 1;
                    continue;
                }
            };
            match result {
                Ok((Ok((judged, reused)), assigned_judge_id, judge_idx, finished_at)) => {
                    let tw = &judged.result;
                    let actual_temperature = judged.temperature;
                    let criterion = &criteria[criterion_assignments[lineup_result_idx]];
                    let entry = &mut judge_last_finish[judge_idx];
                    if entry.is_none() || finished_at > entry.unwrap() {
                        *entry = Some(finished_at);
                    }
                    // A reused judgement sent no request, so it adds no retries,
                    // responses or tokens.
                    if reused {
                        reused_judgements += 1;
                    } else {
                        total_retries += tw.retries_used;
                        judge_stats[judge_idx].total_responses += 1;
                        if tw.hit_max_tokens {
                            judge_stats[judge_idx].max_tokens_hits += 1;
                        }
                        if let Some(usage) = &tw.usage {
                            judge_stats[judge_idx].input_tokens += usage.prompt_tokens;
                            judge_stats[judge_idx].output_tokens += usage.completion_tokens;
                        }
                    }
                    if let Some(verdict) = &tw.verdict {
                        calls_before_refit += 1;
                        if let Some(ref file_mutex) = save_file {
                            let lineup_titles: Vec<&str> = slot.item_ids
                                .iter()
                                .map(|&id| titles[id as usize].as_str())
                                .collect();
                            let lineup_hashes: Vec<&str> = slot.item_ids
                                .iter()
                                .map(|&id| text_hashes[id as usize].as_str())
                                .collect();
                            let mut line = serde_json::json!({
                                "refit": refits_run,
                                "items": lineup_titles,
                                "item_text_hashes": lineup_hashes,
                                "ranking": verdict.ranking,
                                "place_probs": verdict.place_probs,
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "temperature": actual_temperature,
                                "deliberation": resolved.deliberation_enabled,
                                "reasoning_effort": judge_reasoning_efforts[judge_idx],
                                "provider": judges[judge_idx].provider,
                                "chat_template_kwargs": judges[judge_idx].chat_template_kwargs,
                                "criterion": criterion,
                                "prompt_hash": format!("{:016x}", slot.prompt_hash),
                                "logprobs": logprobs_mode,
                                "retries_used": if reused { 0 } else { tw.retries_used },
                                "hit_max_tokens": tw.hit_max_tokens,
                                "reused": reused,
                            });
                            if let (false, Some(usage)) = (reused, &tw.usage) {
                                line["usage"] = usage.to_json();
                            }
                            if save_prompt_text {
                                line["prompt"] = serde_json::json!(*slot.prompt);
                            }
                            if save_response_text {
                                line["response"] = serde_json::json!(tw.response_text);
                            }
                            if save_reasoning_text {
                                line["reasoning"] = serde_json::json!(tw.reasoning_text);
                            }
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }

                        // Decomposed edges for engine and scoring.
                        let mut edges = lineup_verdict_to_edges(
                            &slot.item_ids,
                            verdict,
                            assigned_judge_id,
                        );
                        temper_edges_in_place(&mut edges, judge_verdict_temperatures[judge_idx]);
                        refit_results.extend(edges);
                    } else {
                        failed_parse += 1;
                        if let Some(ref file_mutex) = failures_file {
                            let lineup_titles: Vec<&str> = slot.item_ids
                                .iter()
                                .map(|&id| titles[id as usize].as_str())
                                .collect();
                            let lineup_hashes: Vec<&str> = slot.item_ids
                                .iter()
                                .map(|&id| text_hashes[id as usize].as_str())
                                .collect();
                            let mut line = serde_json::json!({
                                "refit": refits_run,
                                "items": lineup_titles,
                                "item_text_hashes": lineup_hashes,
                                "judge_model": judge_models[judge_idx],
                                "judge_endpoint": judge_endpoints[judge_idx],
                                "temperature": actual_temperature,
                                "deliberation": resolved.deliberation_enabled,
                                "reasoning_effort": judge_reasoning_efforts[judge_idx],
                                "provider": judges[judge_idx].provider,
                                "chat_template_kwargs": judges[judge_idx].chat_template_kwargs,
                                "criterion": criterion,
                                "prompt_hash": format!("{:016x}", slot.prompt_hash),
                                "logprobs": logprobs_mode,
                                "retries_used": tw.retries_used,
                                "hit_max_tokens": tw.hit_max_tokens,
                                "prompt": *slot.prompt,
                                "response": tw.response_text,
                                "reasoning": tw.reasoning_text,
                            });
                            if let Some(ref usage) = tw.usage {
                                line["usage"] = usage.to_json();
                            }
                            let mut f = file_mutex.lock().unwrap();
                            let _ = writeln!(f, "{}", line);
                            let _ = f.flush();
                        }
                        if resolved.verbose {
                            let lineup_titles: Vec<&str> = slot.item_ids
                                .iter()
                                .map(|&id| titles[id as usize].as_str())
                                .collect();
                            if judges.len() > 1 {
                                eprintln!(
                                    "  Warning: unparseable lineup ranking for {} [{}], skipping",
                                    lineup_titles.join(" / "),
                                    judge_display_names[judge_idx],
                                );
                            } else {
                                eprintln!(
                                    "  Warning: unparseable lineup ranking for {}, skipping",
                                    lineup_titles.join(" / "),
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
                            "  Error [{}]: {e}",
                            judge_display_names[judge_idx],
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
            // Keep the judgements collected before the interrupt: they are
            // already in the JSONL, and they form a spawn-order prefix of the
            // batch, so keeping them adds no speed-selected data. Aborted
            // requests were never attempted, so they do not count toward the
            // "X of Y succeeded" denominator. Per-judge wall-time stats are
            // skipped: a partial collection is not comparable to a full one.
            let aborted: usize = judge_aborted.iter().sum();
            total_judgements += calls_before_refit;
            judgements_done += lineups.len() - aborted;
            engine.record_edges(&refit_results);
            break;
        }

        for (j, finish) in judge_last_finish.iter().enumerate() {
            if let Some(t) = finish {
                judge_stats[j].wall_time_sum += t.duration_since(collection_start).as_secs_f64();
                judge_stats[j].collection_count += 1;
            }
        }

        total_judgements += calls_before_refit;
        judgements_done += lineups.len();

        if resolved.verbose {
            eprintln!(
                "  Completed: {} successful, {} failed ({} edges fed)",
                calls_before_refit, lineups.len() - calls_before_refit, refit_results.len(),
            );
        }

        if calls_before_refit == 0 && !lineups.is_empty() {
            eprintln!(
                "Warning: all {} lineup judgements before refit {} failed. \
                 If your endpoint requires an API key, ensure it is set via \
                 --api-key or api_key_env in your config.",
                lineups.len(), refits_run,
            );
        }

        engine.record_edges(&refit_results);

        early_stop = refit_interim_scoring(
            &mut engine, &item_ids, &interim_scoring_options, &judge_info, resolved,
            &titles, &mut interim_rankings, refits_run, total_judgements, loaded_judgements,
        );
        if early_stop {
            break;
        }
    }

    finish_run(
        resolved, "lineup judgements", &judges, &judge_info, &item_ids, &titles, &text_hashes, &engine,
        RunTotals { total_judgements, loaded_judgements, judgements_done, reused_judgements, total_retries, failed_http, failed_parse, judge_stats },
        early_stop, cancelled.load(Ordering::Relaxed), &interim_rankings,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_save_path_bare_filename() {
        // Path::parent() of "x.jsonl" is Some(""), not None. The empty
        // parent must resolve to "." rather than failing the exists check.
        let resolved = resolve_save_path(Path::new("x.jsonl"), "judgements");
        assert_eq!(resolved, PathBuf::from("x.jsonl"));
    }

    #[test]
    fn test_resolve_save_path_explicit_file_in_existing_dir() {
        let dir = std::env::temp_dir();
        let file = dir.join("nanojudge-resolve-save-path-test.jsonl");
        let resolved = resolve_save_path(&file, "judgements");
        assert_eq!(resolved, file);
    }

    #[test]
    fn test_resolve_save_path_directory_gets_timestamped_name() {
        let dir = std::env::temp_dir();
        let resolved = resolve_save_path(&dir, "judgements");
        assert_eq!(resolved.parent(), Some(dir.as_path()));
        let name = resolved.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("judgements-") && name.ends_with(".jsonl"), "{name}");
    }

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
    fn test_temper_lineup_edges_divides_strength_log_ratios() {
        // Places 0.9 then 0.8 give Luce strengths [0.9, 0.08, 0.02]; tempering
        // each edge must divide the log-ratio of its two strengths by T.
        let verdict = nanojudge_core::LineupVerdict {
            ranking: vec![0, 1, 2],
            place_probs: vec![0.9, 0.8],
        };
        let mut edges = lineup_verdict_to_edges(&[10, 20, 30], &verdict, 0);
        temper_edges_in_place(&mut edges, 4.0);
        let s = [0.9f64, 0.08, 0.02];
        for e in &edges {
            let (i, j) = (e.slot1 as usize, e.slot2 as usize);
            let p = e.category_probs;
            assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
            let expected = (s[i] / s[j]).ln() / 4.0;
            assert!(((p[0] / p[1]).ln() - expected).abs() < 1e-9, "{i} vs {j}");
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

        // Simulate 5 refit intervals of 10 pairs each.
        let mut total = 0;
        for _ in 0..5 {
            total += 10;
            assign_judgements_to_judges(10, &weights, &mut cumulative, total, &mut rng);
        }

        // After 50 total pairs with equal weights, each should have ~16-17
        assert_eq!(cumulative.iter().sum::<usize>(), 50);
        for &count in &cumulative {
            assert!((16..=17).contains(&count), "count {count} not in [16, 17]");
        }
    }

    #[test]
    fn test_cumulative_balancing_uneven_refit_intervals() {
        let weights = vec![0.5, 0.5];
        let mut cumulative = vec![0usize; 2];
        let mut rng = rand::rng();

        // Uneven refit intervals: 3, 3, 3.
        let mut total = 0;
        for _ in 0..3 {
            total += 3;
            assign_judgements_to_judges(3, &weights, &mut cumulative, total, &mut rng);
        }

        // 9 pairs split across 2 judges: should be 4 and 5
        assert_eq!(cumulative.iter().sum::<usize>(), 9);
        assert!(cumulative[0] >= 4 && cumulative[0] <= 5);
        assert!(cumulative[1] >= 4 && cumulative[1] <= 5);
    }

    // Hashes of the fixture items A, B, C (truncated SHA-256 of "item:A" etc.),
    // matching the identity keys the score-side tests use.
    const HASH_A: &str = "34482beefb0cc992";
    const HASH_B: &str = "b0e6004ac03e61d2";
    const HASH_C: &str = "9188835ed6d49e09";

    // Two logprobs-mode pairwise records: A>B and B>C, judged by m@http://e.
    fn seed_fixture() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, r#"{{"refit":0,"item1":"A","item2":"B","item1_text_hash":"{HASH_A}","item2_text_hash":"{HASH_B}","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}}"#).unwrap();
        writeln!(f, r#"{{"refit":0,"item1":"B","item2":"C","item1_text_hash":"{HASH_B}","item2_text_hash":"{HASH_C}","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}}"#).unwrap();
        f
    }

    fn seed_test_engine(num_items: usize) -> RankingEngine {
        let item_ids: Vec<i64> = (0..num_items as i64).collect();
        RankingEngine::new(&item_ids, EngineConfig {
            judgement_distribution: JudgementDistribution::Uniform,
            matchmaking_sharpness: 1.0,
            min_uniform_edges: 1,
            regularization_strength: 0.1,
            seed: Some(0),
        })
    }

    #[test]
    fn test_seed_prior_edges_remaps_onto_run_indices() {
        let f = seed_fixture();
        let mut engine = seed_test_engine(3);
        // Deliberately order this run's items differently from the file's
        // hash-sorted order so the remap has to actually move indices:
        // B -> 0, A -> 1, C -> 2.
        let text_hashes = vec![HASH_B.to_string(), HASH_A.to_string(), HASH_C.to_string()];
        let judge_id = nanojudge_core::judge_hash("http://e", "m");
        let mut temps = HashMap::new();
        temps.insert("m@http://e".to_string(), 1.0);

        let (loaded, _) = seed_prior_edges(&mut engine, f.path(), &text_hashes, 2, &[judge_id], true, &temps, false);

        // Both records load; each pairwise record is one edge.
        assert_eq!(loaded, 2);
        assert_eq!(engine.completed_edges.len(), 2);
        // A>B remaps to (1, 0); B>C remaps to (0, 2).
        let a_over_b = engine.completed_edges.iter().find(|e| e.item1 == 1 && e.item2 == 0);
        let b_over_c = engine.completed_edges.iter().find(|e| e.item1 == 0 && e.item2 == 2);
        assert!(a_over_b.is_some(), "A>B edge should remap to (1,0)");
        assert!(b_over_c.is_some(), "B>C edge should remap to (0,2)");
        // Every seeded edge carries the file judge's id.
        assert!(engine.completed_edges.iter().all(|e| e.judge_id == judge_id));
    }

    #[test]
    fn test_seed_prior_edges_skips_unknown_items() {
        let f = seed_fixture();
        let mut engine = seed_test_engine(2);
        // This run has only A and B; C is absent, so the B>C edge must be
        // dropped while A>B still seeds.
        let text_hashes = vec![HASH_A.to_string(), HASH_B.to_string()];
        let judge_id = nanojudge_core::judge_hash("http://e", "m");
        let mut temps = HashMap::new();
        temps.insert("m@http://e".to_string(), 1.0);

        let (loaded, _) = seed_prior_edges(&mut engine, f.path(), &text_hashes, 2, &[judge_id], true, &temps, false);

        // Only A>B survives; it remaps to (0, 1).
        assert_eq!(loaded, 1);
        assert_eq!(engine.completed_edges.len(), 1);
        assert_eq!(engine.completed_edges[0].item1, 0);
        assert_eq!(engine.completed_edges[0].item2, 1);
    }

    #[test]
    fn test_seed_prior_edges_counts_contributing_lineup_judgements() {
        // Three 3-item lineups; this run has only A, B, C. Lineup 1 seeds all 3
        // of its edges, lineup 2 (A, B, D) seeds only A-B, and lineup 3
        // (A, D, E) seeds nothing. Two judgements contributed, from 4 edges.
        const HASH_D: &str = "dddddddddddddddd";
        const HASH_E: &str = "eeeeeeeeeeeeeeee";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for (names, hashes) in [
            (r#""A","B","C""#, [HASH_A, HASH_B, HASH_C]),
            (r#""A","B","D""#, [HASH_A, HASH_B, HASH_D]),
            (r#""A","D","E""#, [HASH_A, HASH_D, HASH_E]),
        ] {
            writeln!(f, r#"{{"refit":0,"items":[{names}],"item_text_hashes":["{}","{}","{}"],"ranking":[0,1,2],"place_probs":[0.9,0.8],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}}"#, hashes[0], hashes[1], hashes[2]).unwrap();
        }
        let mut engine = seed_test_engine(3);
        let text_hashes = vec![HASH_A.to_string(), HASH_B.to_string(), HASH_C.to_string()];
        let judge_id = nanojudge_core::judge_hash("http://e", "m");
        let mut temps = HashMap::new();
        temps.insert("m@http://e".to_string(), 1.0);

        let (loaded, _) = seed_prior_edges(&mut engine, f.path(), &text_hashes, 3, &[judge_id], true, &temps, false);

        assert_eq!(loaded, 2);
        assert_eq!(engine.completed_edges.len(), 4);
    }

    use std::sync::atomic::AtomicUsize;

    /// A stand-in judgement: `Some` is a verdict, `None` an unparseable reply.
    fn reply(verdict: Option<u8>) -> Result<Judged<Option<u8>>, String> {
        Ok(Judged { result: verdict, temperature: 0.5 })
    }

    async fn judge_counted(
        cell: Option<ReuseCell<Option<u8>>>,
        sends: &AtomicUsize,
        outcome: Result<Judged<Option<u8>>, String>,
    ) -> Result<(Option<u8>, bool), String> {
        judge_or_reuse(cell, Option::is_some, || async {
            sends.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            outcome
        })
        .await
        .map(|(judged, reused)| (judged.result, reused))
    }

    #[tokio::test]
    async fn test_judge_or_reuse_sends_once_per_prompt() {
        let sends = AtomicUsize::new(0);
        let cell = Arc::new(OnceCell::new());
        assert_eq!(judge_counted(Some(cell.clone()), &sends, reply(Some(1))).await, Ok((Some(1), false)));
        // The stored verdict comes back, not this attempt's.
        assert_eq!(judge_counted(Some(cell.clone()), &sends, reply(Some(2))).await, Ok((Some(1), true)));
        assert_eq!(sends.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_judge_or_reuse_without_cell_always_sends() {
        let sends = AtomicUsize::new(0);
        assert_eq!(judge_counted(None, &sends, reply(Some(1))).await, Ok((Some(1), false)));
        assert_eq!(judge_counted(None, &sends, reply(Some(2))).await, Ok((Some(2), false)));
        assert_eq!(sends.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_judge_or_reuse_resends_after_failure_or_no_verdict() {
        let sends = AtomicUsize::new(0);
        let cell = Arc::new(OnceCell::new());
        assert_eq!(judge_counted(Some(cell.clone()), &sends, Err("down".into())).await, Err("down".into()));
        assert_eq!(judge_counted(Some(cell.clone()), &sends, reply(None)).await, Ok((None, false)));
        assert_eq!(judge_counted(Some(cell.clone()), &sends, reply(Some(3))).await, Ok((Some(3), false)));
        assert_eq!(judge_counted(Some(cell.clone()), &sends, reply(Some(4))).await, Ok((Some(3), true)));
        assert_eq!(sends.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_judge_or_reuse_concurrent_copies_wait_for_the_first() {
        let sends = AtomicUsize::new(0);
        let cell = Arc::new(OnceCell::new());
        let (first, second) = tokio::join!(
            judge_counted(Some(cell.clone()), &sends, reply(Some(1))),
            judge_counted(Some(cell.clone()), &sends, reply(Some(2))),
        );
        assert_eq!(first, Ok((Some(1), false)));
        assert_eq!(second, Ok((Some(1), true)));
        assert_eq!(sends.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_judge_or_reuse_concurrent_copy_sends_after_first_fails() {
        let sends = AtomicUsize::new(0);
        let cell = Arc::new(OnceCell::new());
        let (first, second) = tokio::join!(
            judge_counted(Some(cell.clone()), &sends, Err("down".into())),
            judge_counted(Some(cell.clone()), &sends, reply(Some(2))),
        );
        assert_eq!(first, Err("down".into()));
        assert_eq!(second, Ok((Some(2), false)));
        assert_eq!(sends.load(Ordering::SeqCst), 2);
    }

    /// A judge that reuses results with deliberation off, unless `temperature`
    /// is above 0 in text mode.
    fn reuse_judge(temperature: f64, logprobs: bool) -> ResolvedJudge {
        ResolvedJudge {
            endpoint: "http://e".into(),
            model: "m".into(),
            api_key: None,
            temperature,
            temperature_jitter: 0.0,
            presence_penalty: None,
            top_p: None,
            logprobs,
            concurrency: 1,
            weight: 1.0,
            min_logprob_coverage: 0.0,
            verdict_temperature: 1.0,
            max_tokens: 16,
            reasoning_effort: Some("none".into()),
            chat_template_kwargs: None,
            provider: None,
            request_timeout: None,
            judge_id: judge_hash("http://e", "m"),
            display_name: "m".into(),
        }
    }

    /// A saved pair record from `reuse_judge`'s judge with matching settings.
    fn reuse_record(temperature: f64) -> serde_json::Value {
        serde_json::json!({
            "judge_model": "m", "judge_endpoint": "http://e",
            "deliberation": false, "reasoning_effort": "none",
            "provider": null, "chat_template_kwargs": null,
            "prompt_hash": "00000000000000ff", "temperature": temperature,
            "hit_max_tokens": false, "category_probs": [1.0, 0.0],
            "response": "Verdict: Option 1",
        })
    }

    fn reused_keys(records: Vec<serde_json::Value>, judge: ResolvedJudge) -> Vec<(usize, u64)> {
        loaded_reuse_entries(records, &[judge], false, Path::new("f.jsonl"))
            .iter()
            .map(|l| l.key)
            .collect()
    }

    #[test]
    fn test_reuses_results_only_for_unsampled_verdicts() {
        assert!(reuses_results(&reuse_judge(0.0, false), false));
        assert!(reuses_results(&reuse_judge(0.7, true), false));
        assert!(!reuses_results(&reuse_judge(0.7, false), false));
        assert!(!reuses_results(&reuse_judge(0.0, false), true));
        let mut reasoning = reuse_judge(0.0, false);
        reasoning.reasoning_effort = None;
        assert!(!reuses_results(&reasoning, false));
    }

    #[test]
    fn test_loaded_reuse_keeps_matching_judgement() {
        let entries = loaded_reuse_entries(vec![reuse_record(0.0)], &[reuse_judge(0.0, false)], false, Path::new("f.jsonl"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, (0, 0xff));
        assert_eq!(entries[0].temperature, 0.0);
        assert_eq!(entries[0].response_text.as_deref(), Some("Verdict: Option 1"));
        assert_eq!(entries[0].reasoning_text, None);
    }

    #[test]
    fn test_loaded_reuse_skips_changed_settings() {
        for (field, value) in [
            ("deliberation", serde_json::json!(true)),
            ("reasoning_effort", serde_json::json!("low")),
            ("provider", serde_json::json!({"order": ["x"]})),
            ("chat_template_kwargs", serde_json::json!({"enable_thinking": false})),
        ] {
            let mut record = reuse_record(0.0);
            record[field] = value;
            assert!(reused_keys(vec![record], reuse_judge(0.0, false)).is_empty(), "{field}");
        }
    }

    #[test]
    fn test_loaded_reuse_text_mode_needs_temperature_zero() {
        assert!(reused_keys(vec![reuse_record(0.7)], reuse_judge(0.0, false)).is_empty());
        assert_eq!(reused_keys(vec![reuse_record(0.7)], reuse_judge(0.7, true)), vec![(0, 0xff)]);
    }
}
