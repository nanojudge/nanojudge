use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use crate::DEFAULT_BENCHMARK_PAIRS;

/// Output format for the final ranking results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Table,
    Json,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Table => write!(f, "table"),
            OutputFormat::Json => write!(f, "json"),
        }
    }
}

#[derive(Parser)]
#[command(name = "nanojudge", version, about = "Rank items using LLM judgements")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand)]
pub enum Commands {
    /// Run ranking on a list of items
    Rank(RankArgs),
    /// Score a saved JSONL judgements file (no API calls)
    Score(ScoreArgs),
    /// Benchmark an LLM endpoint for throughput, latency, and reliability
    Benchmark(BenchmarkArgs),
    /// Send each judge a few test requests and record how its endpoint behaves
    Probe(ProbeArgs),
    /// Create a default config file at ~/.config/nanojudge/config.toml
    Init,
}

/// Persistent configuration settings — available as both CLI flags and config file entries.
/// These match the config file fields 1:1 (except api_key which is CLI/env only).
#[derive(Parser)]
pub struct ConfigArgs {
    /// Bearer token for the API (also reads OPENAI_API_KEY env var).
    /// Used as fallback for judges without api_key_env.
    #[arg(long)]
    pub api_key: Option<String>,

    /// Enable logprob extraction for continuous win probabilities.
    /// Requires an endpoint that supports logprobs.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub logprobs: Option<bool>,

    /// Average judgements per item. Total budget = ceil(judgements_per_item * num_items / lineup_size).
    #[arg(long)]
    pub judgements_per_item: Option<usize>,

    /// Max concurrent LLM requests
    #[arg(long)]
    pub concurrency: Option<usize>,

    /// Minimum fraction of verdict-token probability mass the top-logprobs must
    /// cover for a logprob-based verdict to be trusted. Must be > 0.0 and <= 1.0.
    /// Default: 0.95.
    #[arg(long)]
    pub min_logprob_coverage: Option<f64>,

    /// Temperature applied to each parsed verdict distribution before scoring:
    /// q^(1/T), which divides every edge's log-odds by T. 1.0 = off; > 1 pulls
    /// overconfident verdicts toward 50/50; < 1 sharpens. Distinct from the
    /// sampling temperature. Must be finite and > 0. Can be overridden per-judge.
    /// Default: 3.0 with deliberation enabled, 1.0 without.
    #[arg(long)]
    pub verdict_temperature: Option<f64>,

    /// Judgement distribution: "uniform" or "top-heavy". Default: top-heavy.
    #[arg(long)]
    pub judgement_distribution: Option<String>,

    /// Number of items in each judged lineup: 2 (default) up to 9.
    /// Above 2, each judgement ranks the whole lineup and the ranking is
    /// decomposed into one edge per pair, so one judgement yields more scoring
    /// evidence. Logprobs mode makes those edges soft; text mode keeps the stated
    /// ranking as hard edges. A custom --prompt-template must name exactly as many
    /// options as the lineup size ($optionA..$optionI).
    #[arg(long)]
    pub lineup_size: Option<usize>,

    /// Top-heavy selection sharpness: each item's uncertainty ratio — min/max
    /// of its chance of being in vs out of the top k (see --selection-top-k);
    /// 1 = a coin flip, near 0 = confidently in or out — is raised
    /// to this power before being used as its pairing weight. Lower = flatter =
    /// more exploration; 1.0 = raw ratios. Must be finite and > 0.
    /// Default: 0.7. Only used with top-heavy.
    #[arg(long)]
    pub selection_sharpness: Option<f64>,

    /// How many top places top-heavy selection aims to settle: judgements go
    /// to the items whose place in or out of the top k is still uncertain. 1
    /// aims at first place; 10 is right for "find the top ten, order within
    /// them doesn't matter". Must be >= 1 and <= number of items. Default: 1.
    /// Only used with top-heavy.
    #[arg(long)]
    pub selection_top_k: Option<usize>,

    /// Number of random draws used to estimate each item's chance of being in
    /// the top k at every refit. More draws give a steadier estimate and cost
    /// proportionally more computation. Must be >= 1. Default: 1000. Only used
    /// with top-heavy.
    #[arg(long)]
    pub selection_draws: Option<usize>,

    /// Top-heavy selection cutoff: an item needs at least this uncertainty
    /// ratio to stay a candidate; items below are dropped (the two highest are
    /// always kept). In [0, 1); 0 disables the cutoff. Default: 0 (no cutoff —
    /// sharpness does the shaping). Only used with top-heavy.
    #[arg(long)]
    pub cutoff: Option<f64>,

    /// Top-heavy coverage pull (proportional-fair). Each item's weight is divided
    /// by edge-count^coverage, pulling its cumulative edge count toward
    /// its ratio-implied share. 0 = off, 1 = standard proportional-fair, > 1
    /// over-corrects toward equal coverage. Must be finite and >= 0. Default: 1.
    /// Only used with top-heavy.
    #[arg(long)]
    pub coverage: Option<f64>,

    /// Early stop for top-heavy runs: after each interim fit, end the run once
    /// the probability that every item is on its likelier side of the top-k
    /// boundary (in or out) — the product of the per-item side probabilities —
    /// reaches this value; final
    /// scoring then runs on the judgements collected so far. In (0.5, 1.0),
    /// e.g. 0.95. No default: when absent, the run always uses its full
    /// budget. Rejected with the uniform distribution.
    #[arg(long)]
    pub stop_confidence: Option<f64>,

    /// Max retries per judgement on HTTP errors. Default: 3. Set to 0 to disable.
    #[arg(long)]
    pub retries: Option<usize>,

    /// How much deliberation the LLM should write before its verdict.
    /// Default: "2 paragraphs". Examples: "3 sentences", "1 paragraph", "5 sentences".
    #[arg(long)]
    pub deliberation_length: Option<String>,

    /// Enable LLM deliberation before the verdict. Default: true.
    /// Use --deliberation false to skip deliberation (faster/cheaper, forces max_tokens down to fit just the verdict).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub deliberation: Option<bool>,

    /// Path to a custom prompt template file.
    /// Pairwise templates must contain $criterion, $option1 and $option2;
    /// $name1, $name2 and $length are optional. Lineup templates must contain
    /// $criterion and one $option<letter> per item ($optionA through $optionI);
    /// $length is optional.
    #[arg(long)]
    pub prompt_template: Option<PathBuf>,

    /// Confidence interval level (e.g. 0.95 for 95%). Default: 0.95.
    #[arg(long)]
    pub confidence_level: Option<f64>,

    /// Ghost player regularization strength. Default: 0.01.
    #[arg(long)]
    pub regularization_strength: Option<f64>,

    /// Positional bias prior in probability space (0.0-1.0 exclusive).
    /// 0.5 = no bias (default). >0.5 = model tends to favor item listed first.
    #[arg(long)]
    pub bias_prior: Option<f64>,

    /// Info-gain exponent for matchmaking. Higher = more exploitation. Default: 1.0.
    #[arg(long)]
    pub matchmaking_sharpness: Option<f64>,

    /// Minimum edges per item before the top-heavy distribution kicks in.
    /// Must be >= 1. Default: 2.
    #[arg(long)]
    pub min_uniform_edges: Option<usize>,

    /// Number of judgement attempts to schedule before refitting the scoring model.
    /// Smaller values refit more often (more adaptive pairing, more overhead).
    /// The value is literal, including during uniform pairing. Default: the
    /// number of judgements needed for every item to appear at least once.
    #[arg(long)]
    pub judgements_per_refit: Option<usize>,

    /// Prior variance on log-strengths. Default: 10.0.
    #[arg(long)]
    pub prior_tau2: Option<f64>,

    /// Prior variance on positional bias (logit space). Default: 2.0.
    #[arg(long)]
    pub bias_prior_tau2: Option<f64>,

    /// Print a live ranking table after each refit.
    /// With no value: prints all items. With a number: prints top N only.
    #[arg(long, num_args = 0..=1, default_missing_value = "0")]
    pub live_top: Option<usize>,

    /// Emit the interim ranking after every refit as an `interim_rankings` array
    /// in JSON output. Forces an interim scoring pass each refit even for uniform.
    /// Intended for the benchmark harness to plot convergence.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub emit_interim_rankings: Option<bool>,

    /// Save successful judgements to a JSONL file.
    /// Bare flag: saves to judgements-{timestamp}.jsonl in the current directory.
    /// With a path: saves to that file (or auto-generates a name if path is a directory).
    /// By default saves metadata and verdicts but no text; add --save-prompt-text,
    /// --save-response-text and --save-reasoning-text to include it.
    #[arg(long, num_args = 0..=1, default_missing_value = ".")]
    pub save_successful_judgements: Option<PathBuf>,

    /// Include the prompt sent to the LLM in saved successful judgements.
    /// Requires --save-successful-judgements.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub save_prompt_text: Option<bool>,

    /// Include the LLM's visible answer (deliberation and verdict) in saved
    /// successful judgements. Requires --save-successful-judgements.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub save_response_text: Option<bool>,

    /// Include the model's own reasoning text in saved successful judgements.
    /// Requires --save-successful-judgements.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub save_reasoning_text: Option<bool>,

    /// Output format for final results: "table" or "json". Default: table.
    #[arg(long, value_enum)]
    pub output_format: Option<OutputFormat>,

    /// Show progress during execution.
    #[arg(short, long, num_args = 0..=1, default_missing_value = "true")]
    pub verbose: Option<bool>,

    /// Save failed judgements (unparseable responses) to a JSONL file.
    /// Bare flag: saves to failures-{timestamp}.jsonl in the current directory.
    /// With a path: saves to that file (or auto-generates a name if path is a directory).
    /// Always includes the prompt, response and reasoning text for debugging.
    #[arg(long, num_args = 0..=1, default_missing_value = ".")]
    pub save_failed_judgements: Option<PathBuf>,

    /// RNG seed for reproducible runs. Omit for OS entropy.
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Parser)]
pub struct ScoreArgs {
    /// Path to a JSONL file of saved successful judgements.
    pub file: PathBuf,

    /// Confidence interval level (e.g. 0.95 for 95%). Default: 0.95.
    #[arg(long)]
    pub confidence_level: Option<f64>,

    /// Ghost player regularization strength. Default: 0.01.
    #[arg(long)]
    pub regularization_strength: Option<f64>,

    /// Prior variance on log-strengths. Default: 10.0.
    #[arg(long)]
    pub prior_tau2: Option<f64>,

    /// Prior variance on positional bias (logit space). Default: 2.0.
    #[arg(long)]
    pub bias_prior_tau2: Option<f64>,

    /// Positional bias prior in probability space (0.0-1.0 exclusive).
    /// 0.5 = no bias (default). >0.5 = model tends to favor item listed first.
    #[arg(long)]
    pub bias_prior: Option<f64>,

    /// Temperature applied to each verdict distribution before scoring:
    /// q^(1/T), which divides every edge's log-odds by T. 1.0 = off;
    /// values above 1 pull overconfident verdicts toward 50/50; below 1
    /// sharpens. Default: 3.0 with deliberation, 1.0 without (inferred from
    /// the JSONL file's deliberation field). Must be finite and > 0.
    /// Overridden per-judge by --judge-verdict-temperature.
    #[arg(long)]
    pub verdict_temperature: Option<f64>,

    /// Per-judge verdict temperature override (repeatable). Format:
    /// MODEL@ENDPOINT=TEMPERATURE. Overrides --verdict-temperature for
    /// edges from the matching judge. Must be finite and > 0.
    #[arg(long)]
    pub judge_verdict_temperature: Vec<String>,

    /// Output format for final results: "table" or "json". Default: table.
    #[arg(long, value_enum)]
    pub output_format: Option<OutputFormat>,
}

#[derive(Parser)]
pub struct BenchmarkArgs {
    #[command(flatten)]
    pub cfg: ConfigArgs,

    /// Number of judgement pairs to run (each pair runs both directions)
    #[arg(short, long, default_value = DEFAULT_BENCHMARK_PAIRS)]
    pub num_pairs: usize,

    /// Skip the endpoint probe that normally checks each judge before the run.
    #[arg(long)]
    pub skip_probe: bool,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    pub config: Option<PathBuf>,
}

#[derive(Parser)]
pub struct ProbeArgs {
    #[command(flatten)]
    pub cfg: ConfigArgs,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    pub config: Option<PathBuf>,
}

#[derive(Parser)]
pub struct RankArgs {
    #[command(flatten)]
    pub cfg: ConfigArgs,

    /// The judgement criterion (e.g. "Which is more rewatchable?")
    #[arg(long, required_unless_present = "criterion_file")]
    pub criterion: Option<String>,

    /// File containing one or more criteria separated by ---CRITERION---
    #[arg(long, conflicts_with = "criterion")]
    pub criterion_file: Option<PathBuf>,

    /// File with one item per line, or a directory of text files (each file = one item)
    #[arg(long)]
    pub items: Option<PathBuf>,

    /// Inline item (repeatable)
    #[arg(long = "item")]
    pub inline_items: Vec<String>,

    /// Load a previously saved successful-judgements JSONL file and seed its
    /// edges before collecting new ones, so new pairings and the final ranking
    /// build on the prior run. The file must match this run's lineup size and
    /// logprobs mode, and every judge in it must be a configured judge here (add
    /// a judge with weight 0 to reuse its data without drawing new comparisons
    /// from it). The budget still refers to new judgements only.
    #[arg(long)]
    pub load_judgements: Option<PathBuf>,

    /// Skip the endpoint probe that normally checks each judge before the run.
    #[arg(long)]
    pub skip_probe: bool,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    pub config: Option<PathBuf>,

}
