use clap::Parser;
use std::path::PathBuf;

use crate::DEFAULT_BENCHMARK_PAIRS;

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
    /// Converted to rounds by dividing by comparisons-per-round, rounded down.
    #[arg(long, conflicts_with = "rounds")]
    pub comparisons: Option<usize>,

    /// Max concurrent LLM requests
    #[arg(long)]
    pub concurrency: Option<usize>,

    /// Win probability assigned to a "narrow win" verdict (B or D on the likert scale).
    /// Must be > 0.5 and < 1.0. Default: 0.8. "Clear win" (A/E) is always 1.0/0.0.
    #[arg(long)]
    pub narrow_win: Option<f64>,

    /// Pairing strategy: "balanced" or "top-heavy"
    #[arg(long)]
    pub strategy: Option<String>,

    /// How many top positions to track for the top-heavy strategy.
    /// Default: sqrt(n) * 3, clamped to n-1. Only used with --strategy top-heavy.
    #[arg(long)]
    pub top_k: Option<usize>,

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
    /// The template must contain: $criterion, $option1, $option2, $length
    #[arg(long)]
    pub prompt_template: Option<PathBuf>,

    /// Confidence interval level (e.g. 0.95 for 95%). Default: 0.95.
    #[arg(long)]
    pub confidence_level: Option<f64>,

    /// Ghost player regularization strength. Default: 0.01.
    #[arg(long)]
    pub regularization_strength: Option<f64>,

    /// Number of post-burn-in MCMC iterations for final scoring. Default: 2000.
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

    /// Minimum games per item before using top-heavy strategy. Default: 3.
    #[arg(long)]
    pub min_games: Option<usize>,

    /// Prior variance on log-strengths. Default: 10.0.
    #[arg(long)]
    pub prior_tau2: Option<f64>,

    /// Observation noise variance. Default: 1.0.
    #[arg(long)]
    pub sigma2: Option<f64>,

    /// MH proposal step size for strengths. Default: 0.3.
    #[arg(long)]
    pub proposal_std: Option<f64>,

    /// Prior variance on positional bias (logit space). Default: 2.0.
    #[arg(long)]
    pub bias_prior_tau2: Option<f64>,

    /// MH proposal step size for bias. Default: 0.15.
    #[arg(long)]
    pub bias_proposal_std: Option<f64>,

    /// Prior variance on log-decisiveness (logprobs mode). Default: 1.0.
    #[arg(long)]
    pub decisiveness_prior_tau2: Option<f64>,

    /// MH proposal step size for log-decisiveness (logprobs mode). Default: 0.1.
    #[arg(long)]
    pub decisiveness_proposal_std: Option<f64>,

    /// Print a live ranking table after each round.
    /// With no value: prints all items. With a number: prints top N only.
    #[arg(long, num_args = 0..=1, default_missing_value = "0")]
    pub live_top: Option<usize>,

    /// Save all comparisons to a JSONL file.
    /// Bare flag: saves to comparisons-{timestamp}.jsonl in the current directory.
    /// With a path: saves to that file (or auto-generates a name if path is a directory).
    #[arg(long, num_args = 0..=1, default_missing_value = ".")]
    pub save_comparisons: Option<PathBuf>,

    /// Output JSON instead of table.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub json: Option<bool>,

    /// Show progress during execution.
    #[arg(short, long, num_args = 0..=1, default_missing_value = "true")]
    pub verbose: Option<bool>,

    /// Save failed comparisons (unparseable responses) to a JSONL file.
    /// Bare flag: saves to failures-{timestamp}.jsonl in the current directory.
    /// With a path: saves to that file (or auto-generates a name if path is a directory).
    #[arg(long, num_args = 0..=1, default_missing_value = ".")]
    pub save_failures: Option<PathBuf>,
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
