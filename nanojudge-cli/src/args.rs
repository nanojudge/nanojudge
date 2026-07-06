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
#[command(name = "nanojudge", version, about = "Rank items using LLM pairwise comparisons")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand)]
pub enum Commands {
    /// Run pairwise ranking on a list of items
    Rank(RankArgs),
    /// Benchmark an LLM endpoint for throughput, latency, and reliability
    Benchmark(BenchmarkArgs),
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

    /// Number of comparison rounds
    #[arg(long, conflicts_with = "comparisons")]
    pub rounds: Option<usize>,

    /// Target number of comparisons (alternative to --rounds).
    /// Converted to rounds by dividing by comparisons-per-round, rounded up.
    #[arg(long, conflicts_with = "rounds")]
    pub comparisons: Option<usize>,

    /// Max concurrent LLM requests
    #[arg(long)]
    pub concurrency: Option<usize>,

    /// Minimum fraction of A-D probability mass the top-logprobs must cover for a
    /// logprob-based verdict to be trusted. Must be > 0.0 and <= 1.0. Default: 0.95.
    #[arg(long)]
    pub min_logprob_coverage: Option<f64>,

    /// Comparison distribution: "uniform" or "top-heavy"
    #[arg(long)]
    pub comparison_distribution: Option<String>,

    /// Top-heavy selection sharpness: each item's uncertainty ratio around the
    /// anchor — min/max of the posterior area above vs below the anchor's mean,
    /// 1 = straddles the anchor, near 0 = confidently on one side — is raised
    /// to this power before being used as its pairing weight. Lower = flatter =
    /// more exploration; 1.0 = raw ratios. Must be finite and > 0.
    /// Default: 0.7. Only used with top-heavy.
    #[arg(long)]
    pub selection_sharpness: Option<f64>,

    /// Top-heavy anchor rank, 0-based into the ranking sorted best-first: 0.0
    /// anchors selection on the current leader, 9.0 on the 10th-best (right for
    /// "find the top ten, order within them doesn't matter"). Fractional values
    /// interpolate between adjacent ranks. Must be finite, >= 0, and <= number
    /// of items - 1. Default: 0. Only used with top-heavy.
    #[arg(long)]
    pub anchor_index: Option<f64>,

    /// Top-heavy selection cutoff: an item needs at least this uncertainty
    /// ratio to stay a candidate; items below are dropped (the two highest are
    /// always kept). In [0, 1); 0 disables the cutoff. Default: 0 (no cutoff —
    /// sharpness does the shaping). Only used with top-heavy.
    #[arg(long)]
    pub cutoff: Option<f64>,

    /// Top-heavy coverage pull (proportional-fair). Each item's weight is divided
    /// by games-played^coverage, pulling its cumulative comparison count toward
    /// its ratio-implied share. 0 = off, 1 = standard proportional-fair, > 1
    /// over-corrects toward equal coverage. Must be finite and >= 0. Default: 1.
    /// Only used with top-heavy.
    #[arg(long)]
    pub coverage: Option<f64>,

    /// Top-heavy selection target blend: the prior-predicted anchor strength
    /// counts as this many pseudo-games against the observed anchor's game
    /// count, so the prediction dominates early and fades as the anchor plays
    /// more. Higher = trust the prediction longer. 0 disables the blend (pure
    /// observed anchor). Must be finite and >= 0. Default: 5. Only used with
    /// top-heavy.
    #[arg(long)]
    pub target_prior_games: Option<f64>,

    /// Max retries per comparison on HTTP errors. Default: 3. Set to 0 to disable.
    #[arg(long)]
    pub retries: Option<usize>,

    /// How much analysis the LLM should write before its verdict.
    /// Default: "2 paragraphs". Examples: "3 sentences", "1 paragraph", "5 sentences".
    #[arg(long)]
    pub analysis_length: Option<String>,

    /// Enable LLM reasoning/analysis before the verdict. Default: true.
    /// Use --reasoning false to skip analysis (faster/cheaper, forces max_tokens to 16).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub reasoning: Option<bool>,

    /// Path to a custom prompt template file.
    /// The template must contain: $criterion, $option1, $option2
    #[arg(long)]
    pub prompt_template: Option<PathBuf>,

    /// Confidence interval level (e.g. 0.95 for 95%). Default: 0.95.
    #[arg(long)]
    pub confidence_level: Option<f64>,

    /// Ghost player regularization strength. Default: 0.01.
    #[arg(long)]
    pub regularization_strength: Option<f64>,

    /// Inference engine: "laplace-linear" (deterministic MAP via Newton-CG +
    /// diagonal-Fisher std, O(#comparisons), default) or "mcmc" (Gaussian-BT
    /// sampler).
    #[arg(long)]
    pub inference: Option<String>,

    /// Number of post-burn-in MCMC iterations for final scoring. Default: 5000.
    #[arg(long)]
    pub mcmc_iterations: Option<usize>,

    /// MCMC burn-in iterations for final scoring. Default: 500.
    #[arg(long)]
    pub mcmc_burn_in: Option<usize>,

    /// Positional bias prior in probability space (0.0-1.0 exclusive).
    /// 0.5 = no bias (default). >0.5 = model tends to favor item listed first.
    #[arg(long)]
    pub bias_prior: Option<f64>,

    /// Info-gain exponent for matchmaking. Higher = more exploitation. Default: 1.0.
    #[arg(long)]
    pub matchmaking_sharpness: Option<f64>,

    /// Minimum games per item before the top-heavy distribution kicks in. Default: 2.
    #[arg(long)]
    pub min_uniform_games: Option<usize>,

    /// Number of scoring refits per round once top-heavy pairing is active.
    /// 1 (default) refits once per round, as before. Higher values split each
    /// round's pairs into that many chunks, refitting strengths/CIs and
    /// re-deriving selection weights after each chunk — more adaptive pairing.
    /// The uniform stage is never subdivided. Must be >= 1.
    #[arg(long)]
    pub refits_per_round: Option<usize>,

    /// Prior variance on log-strengths. Default: 10.0.
    #[arg(long)]
    pub prior_tau2: Option<f64>,

    /// MH proposal step size for strengths. Default: 0.3.
    #[arg(long)]
    pub proposal_std: Option<f64>,

    /// Prior variance on positional bias (logit space). Default: 2.0.
    #[arg(long)]
    pub bias_prior_tau2: Option<f64>,

    /// MH proposal step size for bias. Default: 0.15.
    #[arg(long)]
    pub bias_proposal_std: Option<f64>,

    /// Print a live ranking table after each round.
    /// With no value: prints all items. With a number: prints top N only.
    #[arg(long, num_args = 0..=1, default_missing_value = "0")]
    pub live_top: Option<usize>,

    /// Emit the interim ranking after every round as a `round_rankings` array
    /// in JSON output. Forces an interim MCMC pass each round even for uniform.
    /// Intended for the benchmark harness to plot per-round convergence.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub emit_round_rankings: Option<bool>,

    /// Save all comparisons to a JSONL file.
    /// Bare flag: saves to comparisons-{timestamp}.jsonl in the current directory.
    /// With a path: saves to that file (or auto-generates a name if path is a directory).
    #[arg(long, num_args = 0..=1, default_missing_value = ".")]
    pub save_comparisons: Option<PathBuf>,

    /// Output format for final results: "table" or "json". Default: table.
    #[arg(long, value_enum)]
    pub output_format: Option<OutputFormat>,

    /// Show progress during execution.
    #[arg(short, long, num_args = 0..=1, default_missing_value = "true")]
    pub verbose: Option<bool>,

    /// Save failed comparisons (unparseable responses) to a JSONL file.
    /// Bare flag: saves to failures-{timestamp}.jsonl in the current directory.
    /// With a path: saves to that file (or auto-generates a name if path is a directory).
    #[arg(long, num_args = 0..=1, default_missing_value = ".")]
    pub save_failures: Option<PathBuf>,

    /// RNG seed for reproducible runs. Omit for OS entropy.
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Parser)]
pub struct BenchmarkArgs {
    #[command(flatten)]
    pub cfg: ConfigArgs,

    /// Number of comparison pairs to run (each pair runs both directions)
    #[arg(short, long, default_value = DEFAULT_BENCHMARK_PAIRS)]
    pub num_pairs: usize,

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    pub config: Option<PathBuf>,
}

#[derive(Parser)]
pub struct RankArgs {
    #[command(flatten)]
    pub cfg: ConfigArgs,

    /// The comparison criterion (e.g. "Which is more rewatchable?")
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

    /// Path to config file (default: ~/.config/nanojudge/config.toml)
    #[arg(long)]
    pub config: Option<PathBuf>,

}
