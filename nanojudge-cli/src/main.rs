mod benchmark;
mod config;
mod llm;
mod output;
mod parse;
mod prompt;

use clap::Parser;
use nanojudge_core::{
    ComparisonInput, EngineConfig, RankingEngine, ScoringOptions, Strategy,
    calculate_total_expected_comparisons, run_scoring,
};
use reqwest::Client;
use std::collections::HashSet;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

use crate::llm::{LlmConfig, compare_pair};

const DEFAULT_CONCURRENCY: usize = 32;
const DEFAULT_TEMPERATURE_JITTER: f64 = 0.0;
const DEFAULT_MAX_RETRIES: usize = 3;
const DEFAULT_ANALYSIS_LENGTH: &str = "2 paragraphs";
const DEFAULT_BENCHMARK_PAIRS: &str = "100";

pub fn bail(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

#[derive(Parser)]
#[command(name = "nanojudge", version, about = "Rank items using LLM pairwise comparisons")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Run pairwise ranking on a list of items
    Rank(RankArgs),
    /// Benchmark an LLM endpoint for throughput, latency, and reliability
    Benchmark(BenchmarkArgs),
    /// Create a default config file at ~/.config/nanojudge/config.toml
    Init,
}

#[derive(Parser)]
struct BenchmarkArgs {
    /// OpenAI-compatible base URL (e.g. http://localhost:8000)
    #[arg(long)]
    endpoint: Option<String>,

    /// Bearer token for the API (also reads OPENAI_API_KEY env var)
    #[arg(long)]
    api_key: Option<String>,

    /// Model ID for the API
    #[arg(long)]
    model: Option<String>,

    /// Number of comparison pairs to run (each pair runs both directions)
    #[arg(short, long, default_value = DEFAULT_BENCHMARK_PAIRS)]
    num_pairs: usize,

    /// Max concurrent LLM requests
    #[arg(long)]
    concurrency: Option<usize>,

    /// LLM sampling temperature (required — each model needs a different value)
    #[arg(long)]
    temperature: Option<f64>,

    /// Temperature jitter: std dev of N(1.0, jitter) multiplier. 0.0 = no jitter (default).
    #[arg(long)]
    temperature_jitter: Option<f64>,

    /// Presence penalty: penalizes repeated tokens. Range: -2.0 to 2.0.
    #[arg(long)]
    presence_penalty: Option<f64>,

    /// Top-p (nucleus sampling). Range: 0.0 to 1.0.
    #[arg(long)]
    top_p: Option<f64>,

    /// Win probability for narrow wins. Default: 0.8.
    #[arg(long)]
    narrow_win: Option<f64>,

    /// Path to a custom prompt template file
    #[arg(long)]
    prompt_template: Option<PathBuf>,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Parser)]
struct RankArgs {
    /// The comparison criterion (e.g. "Which is more rewatchable?")
    #[arg(long)]
    criterion: String,

    /// File with one item per line, or a directory of text files (each file = one item)
    #[arg(long)]
    items: Option<PathBuf>,

    /// Inline item (repeatable)
    #[arg(long = "item")]
    inline_items: Vec<String>,

    /// OpenAI-compatible base URL (e.g. http://localhost:8000)
    #[arg(long)]
    endpoint: Option<String>,

    /// Bearer token for the API (also reads OPENAI_API_KEY env var)
    #[arg(long)]
    api_key: Option<String>,

    /// Model ID for the API
    #[arg(long)]
    model: Option<String>,

    /// Number of comparison rounds
    #[arg(long)]
    rounds: Option<usize>,

    /// Max concurrent LLM requests
    #[arg(long)]
    concurrency: Option<usize>,

    /// Output JSON instead of table
    #[arg(long)]
    json: bool,

    /// Show progress during execution
    #[arg(short, long)]
    verbose: bool,

    /// Pairing strategy: "balanced" or "top-heavy"
    #[arg(long)]
    strategy: Option<String>,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Save a sample of comparisons to JSONL for inspection.
    /// Integer (e.g. 50) = exact count, float (e.g. 0.3) = fraction of total.
    #[arg(long)]
    save_comparisons: Option<String>,

    /// Output path for saved comparisons (default: comparisons.jsonl)
    #[arg(long)]
    save_comparisons_to: Option<PathBuf>,

    /// Win probability assigned to a "narrow win" verdict (B or D on the likert scale).
    /// Must be > 0.5 and < 1.0. Default: 0.8. "Clear win" (A/E) is always 1.0/0.0.
    #[arg(long)]
    narrow_win: Option<f64>,

    /// LLM sampling temperature (required — each model needs a different value)
    #[arg(long)]
    temperature: Option<f64>,

    /// Temperature jitter: std dev of N(1.0, jitter) multiplier. 0.0 = no jitter (default).
    #[arg(long)]
    temperature_jitter: Option<f64>,

    /// Presence penalty: penalizes repeated tokens. Range: -2.0 to 2.0.
    #[arg(long)]
    presence_penalty: Option<f64>,

    /// Top-p (nucleus sampling). Range: 0.0 to 1.0.
    #[arg(long)]
    top_p: Option<f64>,

    /// How much analysis the LLM should write before its verdict.
    /// Default: "2 paragraphs". Examples: "3 sentences", "1 paragraph", "5 sentences".
    #[arg(long)]
    analysis_length: Option<String>,

    /// How many top positions to track for the top-heavy strategy.
    /// Default: sqrt(n) * 3, clamped to n-1 — a rough heuristic with no empirical backing,
    /// just a guess at how many top items users typically care about.
    /// Only used with --strategy top-heavy.
    #[arg(long)]
    top_k: Option<usize>,

    /// Max retries per comparison on HTTP errors. Default: 3. Set to 0 to disable.
    #[arg(long)]
    retries: Option<usize>,

    /// Path to a custom prompt template file.
    /// The template must contain: $criterion, $option1, $option2, $length
    #[arg(long)]
    prompt_template: Option<PathBuf>,
}

const TITLE_MAX_LEN: usize = 20;

/// Derive a display title from item text: first 20 chars, hard cut.
fn title_from_text(text: &str) -> String {
    if text.chars().count() <= TITLE_MAX_LEN {
        text.to_string()
    } else {
        text.chars().take(TITLE_MAX_LEN).collect()
    }
}

/// Parse a string as either a JSON array of strings or plain text (one item per line).
fn parse_items_from_str(content: &str) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.starts_with('[') {
        // Try JSON array
        let items: Vec<String> = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| bail(format!("File looks like JSON but failed to parse: {e}")));
        items.into_iter().filter(|s| !s.trim().is_empty()).collect()
    } else {
        // Plain text, one item per line
        trimmed
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// Parse --save-comparisons value: float with '.' → fraction of total, integer → exact count.
fn parse_save_count(value: &str, total: usize) -> usize {
    if value.contains('.') {
        let frac: f64 = value.parse()
            .unwrap_or_else(|_| bail(format!("Invalid fraction for --save-comparisons: \"{value}\"")));
        if !(0.0..=1.0).contains(&frac) {
            bail(format!("--save-comparisons fraction must be between 0.0 and 1.0, got {frac}"));
        }
        (frac * total as f64).round() as usize
    } else {
        let count: usize = value.parse()
            .unwrap_or_else(|_| bail(format!("Invalid count for --save-comparisons: \"{value}\"")));
        count.min(total)
    }
}

/// Plain text file extensions that we read from directories.
const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "html", "csv", "json", "xml", "rst", "log", "yaml", "yml", "toml",
];

/// Load items from all sources: --items file/dir, --item inline args, or stdin.
/// Returns (titles, texts) where titles are for display and texts are sent to the LLM.
fn load_items(args: &RankArgs) -> (Vec<String>, Vec<String>) {
    let mut titles = Vec::new();
    let mut texts = Vec::new();

    if let Some(ref path) = args.items {
        if path.is_dir() {
            // Directory mode: each file is an item
            let entries = std::fs::read_dir(path)
                .unwrap_or_else(|e| bail(format!("Failed to read directory {}: {e}", path.display())));

            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .collect();

            // Sort by filename for deterministic ordering
            files.sort_by_key(|e| e.file_name());

            let mut skipped = 0usize;
            let total = files.len();

            for entry in &files {
                let file_path = entry.path();
                let ext = file_path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");

                if !TEXT_EXTENSIONS.contains(&ext) {
                    skipped += 1;
                    continue;
                }

                let content = std::fs::read_to_string(&file_path)
                    .unwrap_or_else(|e| bail(format!("Failed to read {}: {e}", file_path.display())));
                let content = content.trim().to_string();

                if content.is_empty() {
                    skipped += 1;
                    continue;
                }

                let stem = file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unnamed")
                    .to_string();

                titles.push(stem);
                texts.push(content);
            }

            let loaded = titles.len();
            eprintln!("Found {total} files, loaded {loaded}, skipped {skipped} (unsupported format or empty)");
        } else {
            // File mode: one item per line or JSON array
            let content = std::fs::read_to_string(path)
                .unwrap_or_else(|e| bail(format!("Failed to read items file {}: {e}", path.display())));
            texts = parse_items_from_str(&content);
            titles = texts.iter().map(|t| title_from_text(t)).collect();
        }
    }

    // From inline --item flags
    for item in &args.inline_items {
        titles.push(title_from_text(item));
        texts.push(item.clone());
    }

    // From stdin (only if no file/dir and no inline items)
    if texts.is_empty() {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            bail("No items provided. Use --items <file|dir>, --item <name>, or pipe items via stdin.");
        }
        let content: String = stdin.lock().lines()
            .map(|l| l.expect("Failed to read from stdin"))
            .collect::<Vec<_>>()
            .join("\n");
        texts = parse_items_from_str(&content);
        titles = texts.iter().map(|t| title_from_text(t)).collect();
    }

    if texts.len() < 2 {
        bail(format!("Need at least 2 items to rank, got {}", texts.len()));
    }
    (titles, texts)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Rank(args) => run_rank(args).await,
        Commands::Benchmark(args) => run_benchmark_cmd(args).await,
        Commands::Init => {
            let path = config::create_default_config();
            println!("Created config at {}", path.display());
            println!("Edit it to set your default endpoint, model, etc.");
        }
    }
}

async fn run_benchmark_cmd(args: BenchmarkArgs) {
    let config_path = args.config.clone().unwrap_or_else(config::config_path);
    let cfg = config::load_config(&config_path);

    let endpoint = args.endpoint.clone()
        .or(cfg.endpoint)
        .unwrap_or_else(|| {
            bail(format!("No endpoint specified. Pass --endpoint or set it in {}", config_path.display()));
        });
    let model = args.model.clone()
        .or(cfg.model)
        .unwrap_or_else(|| {
            bail(format!("No model specified. Pass --model or set it in {}", config_path.display()));
        });
    let concurrency = args.concurrency.or(cfg.concurrency).unwrap_or(DEFAULT_CONCURRENCY);
    let temperature = args.temperature
        .or(cfg.temperature)
        .unwrap_or_else(|| {
            bail(format!("No temperature specified. Pass --temperature or set it in {}", config_path.display()));
        });
    let temperature_jitter = args.temperature_jitter.or(cfg.temperature_jitter).unwrap_or(DEFAULT_TEMPERATURE_JITTER);
    let presence_penalty = args.presence_penalty.or(cfg.presence_penalty);
    let top_p = args.top_p.or(cfg.top_p);
    let narrow_win = args.narrow_win.unwrap_or(parse::DEFAULT_NARROW_WIN);

    let template = {
        let template_path = args.prompt_template.clone()
            .or_else(|| cfg.prompt_template.map(PathBuf::from));
        match template_path {
            Some(path) => prompt::load_template(&path),
            None => prompt::DEFAULT_TEMPLATE.to_string(),
        }
    };

    let api_key = args.api_key.clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    if args.num_pairs == 0 {
        bail("--num-pairs must be at least 1");
    }

    benchmark::run_benchmark(
        &endpoint,
        &model,
        api_key,
        args.num_pairs,
        concurrency,
        temperature,
        temperature_jitter,
        presence_penalty,
        top_p,
        narrow_win,
        &template,
    ).await;
}

async fn run_rank(args: RankArgs) {
    // Load config file, merge with CLI args (CLI wins)
    let config_path = args.config.clone().unwrap_or_else(config::config_path);
    let cfg = config::load_config(&config_path);

    let endpoint = args.endpoint.clone()
        .or(cfg.endpoint)
        .unwrap_or_else(|| {
            bail(format!("No endpoint specified. Pass --endpoint or set it in {}", config_path.display()));
        });
    let model = args.model.clone()
        .or(cfg.model)
        .unwrap_or_else(|| {
            bail(format!("No model specified. Pass --model or set it in {}", config_path.display()));
        });
    let rounds = args.rounds
        .or(cfg.rounds)
        .unwrap_or_else(|| {
            bail(format!("No rounds specified. Pass --rounds or set it in {}", config_path.display()));
        });
    let concurrency = args.concurrency.or(cfg.concurrency).unwrap_or(DEFAULT_CONCURRENCY);

    // Load prompt template: CLI arg > config file > built-in default
    let prompt_template = {
        let template_path = args.prompt_template.clone()
            .or_else(|| cfg.prompt_template.map(PathBuf::from));
        match template_path {
            Some(path) => prompt::load_template(&path),
            None => prompt::DEFAULT_TEMPLATE.to_string(),
        }
    };

    let narrow_win = args.narrow_win.unwrap_or(parse::DEFAULT_NARROW_WIN);
    if narrow_win <= 0.5 || narrow_win >= 1.0 {
        bail("--narrow-win must be greater than 0.5 and less than 1.0");
    }

    let (titles, texts) = load_items(&args);
    let item_ids: Vec<i64> = (0..texts.len() as i64).collect();

    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    let temperature = args.temperature
        .or(cfg.temperature)
        .unwrap_or_else(|| {
            bail(format!("No temperature specified. Pass --temperature or set it in {}", config_path.display()));
        });
    let temperature_jitter = args.temperature_jitter.or(cfg.temperature_jitter).unwrap_or(DEFAULT_TEMPERATURE_JITTER);
    let presence_penalty = args.presence_penalty.or(cfg.presence_penalty);
    let top_p = args.top_p.or(cfg.top_p);
    let analysis_length = args.analysis_length.clone().unwrap_or_else(|| DEFAULT_ANALYSIS_LENGTH.to_string());

    let llm_config = Arc::new(LlmConfig {
        endpoint: endpoint.clone(),
        model: model.clone(),
        api_key,
        temperature,
        temperature_jitter,
        presence_penalty,
        top_p,
    });

    let prompt_template = Arc::new(prompt_template);

    let client = Client::new();
    let titles = Arc::new(titles);
    let texts = Arc::new(texts);

    let total_planned = calculate_total_expected_comparisons(texts.len(), rounds);

    if args.verbose {
        eprintln!(
            "Ranking {} items across {} rounds ({} comparisons planned)",
            texts.len(),
            rounds,
            total_planned,
        );
        eprintln!("Criterion: \"{}\"", args.criterion);
        eprintln!("Endpoint: {} | Model: {}", endpoint, model);
    }

    // Set up comparison saving if requested
    let save_file = if let Some(ref save_value) = args.save_comparisons {
        let save_count = parse_save_count(save_value, total_planned);
        let save_path = args.save_comparisons_to.clone()
            .unwrap_or_else(|| PathBuf::from("comparisons.jsonl"));

        let save_indices: HashSet<usize> = if save_count >= total_planned {
            (0..total_planned).collect()
        } else {
            use rand::seq::index::sample;
            let mut rng = rand::rng();
            sample(&mut rng, total_planned, save_count).into_iter().collect()
        };

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&save_path)
            .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", save_path.display())));

        if args.verbose {
            eprintln!("Saving {} comparisons to {}", save_count, save_path.display());
        }

        Some((std::sync::Mutex::new(file), save_indices))
    } else {
        None
    };

    let mut global_idx: usize = 0;

    let strategy = match args.strategy.as_deref() {
        Some("balanced") | None => Strategy::Balanced,
        Some("top-heavy") => Strategy::TopHeavy,
        Some(other) => bail(format!("Unknown strategy \"{other}\". Use \"balanced\" or \"top-heavy\".")),
    };

    if args.top_k.is_some() && matches!(strategy, Strategy::Balanced) {
        eprintln!("Warning: --top-k has no effect with the balanced strategy. It only applies to --strategy top-heavy.");
    }

    // Pure heuristic — no empirical basis. Just a guess at how many top
    // positions users typically care about for a given list size.
    let top_k = args.top_k.unwrap_or_else(|| {
        ((texts.len() as f64).sqrt() * 3.0) as usize
    }).min(texts.len() - 1);

    let engine_config = EngineConfig {
        strategy,
        matchmaking_sharpness: 1.0,
        min_games_before_strategy: 3,
        number_of_rounds: Some(rounds),
    };
    let mut engine = RankingEngine::new(&item_ids, engine_config);

    let max_retries = args.retries.unwrap_or(DEFAULT_MAX_RETRIES);

    let mut total_comparisons: usize = 0;
    let mut total_retries: usize = 0;
    let mut failed_http: usize = 0;
    let mut failed_parse: usize = 0;

    for round in 0..rounds {
        let pairs = engine.generate_pairs_for_round(round);

        if args.verbose {
            eprintln!("Round {}/{}: {} pairs", round + 1, rounds, pairs.len());
        }

        // Run comparisons with bounded concurrency
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut handles = Vec::with_capacity(pairs.len());

        for (id_a, id_b) in &pairs {
            let sem = semaphore.clone();
            let client = client.clone();
            let llm_config = llm_config.clone();
            let texts = texts.clone();
            let criterion = args.criterion.clone();
            let analysis_length = analysis_length.clone();
            let template = prompt_template.clone();
            let id_a = *id_a;
            let id_b = *id_b;

            let verbose = args.verbose;
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                compare_pair(
                    &client,
                    &llm_config,
                    &template,
                    &criterion,
                    &texts[id_a as usize],
                    &texts[id_b as usize],
                    id_a,
                    id_b,
                    narrow_win,
                    &analysis_length,
                    max_retries,
                    verbose,
                )
                .await
            });

            handles.push(handle);
        }

        // Collect results
        let mut round_results: Vec<ComparisonInput> = Vec::new();

        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => {
                    total_retries += result.retries_used;
                    if let Some(p) = result.parse_result.item1_win_probability {
                        // Save to JSONL if this index was selected
                        if let Some((ref file_mutex, ref indices)) = save_file {
                            if indices.contains(&global_idx) {
                                let line = serde_json::json!({
                                    "round": round + 1,
                                    "item1": titles[result.item1_id as usize],
                                    "item2": titles[result.item2_id as usize],
                                    "probability": p,
                                    "response": result.response_text,
                                });
                                let mut f = file_mutex.lock().unwrap();
                                let _ = writeln!(f, "{}", line);
                                let _ = f.flush();
                            }
                        }

                        round_results.push(ComparisonInput {
                            item1: result.item1_id,
                            item2: result.item2_id,
                            item1_win_probability: p,
                        });
                    } else {
                        failed_parse += 1;
                        if args.verbose {
                            eprintln!("  Warning: unparseable response, skipping comparison");
                        }
                    }
                }
                Ok(Err(e)) => {
                    failed_http += 1;
                    if args.verbose {
                        eprintln!("  Error (after exhausting {} retries): {e}", max_retries);
                    }
                }
                Err(e) => {
                    failed_http += 1;
                    if args.verbose {
                        eprintln!("  Task panicked: {e}");
                    }
                }
            }
            global_idx += 1;
        }

        total_comparisons += round_results.len();

        if args.verbose {
            eprintln!(
                "  Completed: {} successful, {} failed",
                round_results.len(),
                pairs.len() - round_results.len(),
            );
        }

        engine.record_results(&round_results);
        engine.update_current_ratings();

        // TopHeavy needs interim MCMC scoring to guide next round's pairing
        if matches!(strategy, Strategy::TopHeavy) && !engine.completed_comparisons.is_empty() {
            let interim = run_scoring(
                &item_ids,
                &engine.completed_comparisons,
                &ScoringOptions {
                    iterations: 200,
                    burn_in: 100,
                    confidence_level: 0.95,
                    top_k,
                    warm_start: None,
                    regularization_strength: 0.01,
                },
            );
            engine.mcmc_top_k_probs = interim.top_k_probs;
            engine.mcmc_sample_means = interim.sample_means;
        }
    }

    if total_comparisons == 0 {
        bail("All comparisons failed. No results to score.");
    }

    if args.verbose {
        eprintln!("Running final MCMC scoring ({total_comparisons} comparisons)...");
    }

    // Final scoring with full MCMC
    let scoring_result = run_scoring(
        &item_ids,
        &engine.completed_comparisons,
        &ScoringOptions {
            iterations: 2000,
            burn_in: 500,
            confidence_level: 0.95,
            top_k: 0,
            warm_start: None,
            regularization_strength: 0.01,
        },
    );

    if args.verbose {
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

    if args.json {
        output::print_json(&scoring_result.rankings, &titles, rounds, total_comparisons);
    } else {
        output::print_table(
            &scoring_result.rankings,
            &titles,
            &engine.games_played,
            rounds,
            total_comparisons,
            scoring_result.positional_bias,
            scoring_result.positional_bias_confidence_interval,
        );
    }
}
