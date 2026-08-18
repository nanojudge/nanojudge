/// Config file loading and creation for nanojudge CLI.
///
/// Config lives at ~/.config/nanojudge/config.toml.
/// All fields are optional — CLI args override config values.
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::args::OutputFormat;
use crate::bail;

/// Per-judge configuration from a [[judge]] TOML block.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    pub endpoint: String,
    pub model: String,
    pub concurrency: Option<usize>,
    pub weight: Option<f64>,
    pub temperature: Option<f64>,
    pub temperature_jitter: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub top_p: Option<f64>,
    pub min_logprob_coverage: Option<f64>,
    /// Temperature applied to this judge's parsed verdict distribution
    /// (q^(1/T)) before scoring. Distinct from the sampling `temperature`.
    pub verdict_temperature: Option<f64>,
    pub api_key_env: Option<String>,
    pub max_tokens: Option<u32>,
    /// OpenRouter extension: controls model reasoning/thinking mode.
    /// Set to "none" to disable chain-of-thought for models like Qwen
    /// that otherwise produce <think>...</think> blocks.
    pub reasoning_effort: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, toml::Value>>,
}

#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub struct NanojudgeConfig {
    pub rounds: Option<usize>,
    pub judgements: Option<usize>,
    pub concurrency: Option<usize>,
    pub prompt_template: Option<String>,
    pub logprobs: Option<bool>,
    pub analysis_length: Option<String>,
    pub judgement_distribution: Option<String>,
    /// Number of items in each judged lineup: 2 or 3.
    pub lineup_size: Option<usize>,
    pub selection_sharpness: Option<f64>,
    pub anchor_index: Option<f64>,
    pub cutoff: Option<f64>,
    pub coverage: Option<f64>,
    pub target_prior_edges: Option<f64>,
    pub stop_confidence: Option<f64>,
    pub retries: Option<usize>,
    pub confidence_level: Option<f64>,
    pub regularization_strength: Option<f64>,
    pub bias_prior: Option<f64>,
    pub matchmaking_sharpness: Option<f64>,
    pub min_uniform_edges: Option<usize>,
    pub refits_per_round: Option<usize>,
    pub prior_tau2: Option<f64>,
    pub bias_prior_tau2: Option<f64>,
    pub reasoning_enabled: Option<bool>,
    pub min_logprob_coverage: Option<f64>,
    pub verdict_temperature: Option<f64>,
    pub live_top: Option<usize>,
    pub save_judgements: Option<PathBuf>,
    pub output_format: Option<OutputFormat>,
    pub verbose: Option<bool>,
    pub save_failures: Option<PathBuf>,
    pub seed: Option<u64>,
    /// Judge panel configuration. At least one [[judge]] block is required.
    pub judge: Option<Vec<JudgeConfig>>,
}

const DEFAULT_CONFIG_TEMPLATE: &str = "\
# nanojudge configuration
# All values here can be overridden by CLI flags unless noted otherwise.

# Number of judgement rounds (mutually exclusive with judgements)
# rounds = 10

# Target number of judgements (alternative to rounds).
# Converted to rounds by dividing by judgements-per-round, rounded up.
# judgements = 500

# Default max concurrent LLM requests per judge (can be overridden per-judge)
# concurrency = 16

# Path to a custom prompt template file.
# The template must contain these variables: $criterion, $option1, $option2
# If not set, the built-in default prompt is used.
# prompt_template = \"/path/to/my-prompt.txt\"

# Enable logprob extraction for continuous win probabilities.
# Requires an endpoint that supports logprobs (e.g. vLLM, OpenAI direct).
# When false, uses text-based verdict parsing (discrete) — may need more rounds.
# logprobs = false

# How much analysis the LLM should write before its verdict.
# Examples: \"3 sentences\", \"1 paragraph\", \"5 sentences\".
# analysis_length = \"2 paragraphs\"

# Whether to have the LLM reason before giving its verdict.
# When false, the LLM skips analysis and outputs only the verdict line.
# Faster and cheaper, but may reduce accuracy. max_tokens is forced to 16.
# Cannot be used with a custom prompt_template.
# reasoning_enabled = true

# Judgement distribution: \"uniform\" or \"top-heavy\".
# Uniform gives equal attention to all items. Top-heavy focuses on contenders.
# judgement_distribution = \"uniform\"

# Number of items in each judged lineup: 2 (default) or 3.
# With 3, each judgement ranks three items and is folded into up to three edges,
# so one LLM call yields more information. Requires logprobs for the
# full benefit.
# lineup_size = 2

# Tuning for the top-heavy distribution. Each item's pairing weight is its
# uncertainty ratio around the anchor: min/max of the probability of sitting
# above vs below the anchor, integrated over both the item's and the anchor's
# posterior uncertainty — 1 when the item straddles the anchor, near 0 once
# it is confidently above or below — raised to `selection_sharpness`. The
# first item of each judgement is drawn from these weights; its opponent is
# picked by info-gain matchmaking from a rating window around it, so the
# contested boundary gets the judgements.
#   selection_sharpness: lower = flatter = more exploration; 1.0 = raw ratios (> 0).
#   anchor_index: which rank is the anchor, 0-based into the ranking sorted
#             best-first. 0.0 = the current leader; 9.0 = the 10th-best (right
#             for \"find the top ten, order within them doesn't matter\").
#             Fractional values interpolate between adjacent ranks: 0.5 targets
#             the midpoint of the 1st and 2nd items. Must be finite, >= 0, and
#             <= number of items - 1.
#   cutoff:   minimum uncertainty ratio to stay a candidate; items below are
#             dropped, except the two highest, which are always kept.
#             [0,1); 0 disables the cutoff.
#   coverage: proportional-fair pull. Each weight is divided by
#             edge-count^coverage, pulling cumulative edges toward the
#             ratio-implied share. 0 = off, 1 = proportional-fair, > 1 over-corrects.
# Only used with judgement_distribution = \"top-heavy\".
# selection_sharpness = 0.7
# anchor_index = 0.0
# cutoff = 0.0
# coverage = 1.0

# Early stop for top-heavy runs. After each interim fit, the run ends early
# once the probability that every item sits on its side of the anchor — the
# product of the per-item side probabilities — reaches this value; final
# scoring then runs on the judgements collected so far. In (0.5, 1.0), e.g.
# 0.95. No default: when unset, the run always uses its full round budget.
# Rejected with judgement_distribution = \"uniform\".
# stop_confidence = 0.95

# Minimum fraction of verdict-token probability mass the top-logprobs must
# cover for a logprob-based verdict to be trusted (logprobs mode only) — the
# option digits for pairwise, the option letters for three-item lineup. Below this, the
# judgement is treated as unparseable. Must be > 0.0 and <= 1.0.
# Can be overridden per-judge.
# min_logprob_coverage = 0.95

# Temperature applied to each parsed verdict distribution before scoring:
# q^(1/T), which divides every edge's log-odds by T. 1.0 = off. > 1 pulls
# overconfident verdicts toward 50/50; < 1 sharpens. Distinct from the
# sampling `temperature` — this tempers the parsed verdict, not the LLM call.
# Text-mode (non-logprob) verdicts are one-hot and unaffected.
# Must be finite and > 0. Can be overridden per-judge.
# Default: 3.0 when reasoning is enabled (a verdict token written after an
# analysis is near-deterministic, so its logprobs read overconfident);
# 1.0 when reasoning is disabled.
# verdict_temperature = 3.0

# Print a live ranking table after each round. 0 = all items, N = top N.
# Omit or comment out to disable.
# live_top = 10

# Save all judgements to a JSONL file. Value is a path.
# Use \".\" for the current directory (auto-generates judgements-{timestamp}.jsonl).
# A directory path auto-generates a filename inside it.
# Omit or comment out to disable.
# save_judgements = \".\"

# Output format for final results: \"table\" or \"json\".
# output_format = \"table\"

# Show progress during execution.
# verbose = false

# Save failed judgements (unparseable responses) to a JSONL file. Value is a path.
# Use \".\" for the current directory (auto-generates failures-{timestamp}.jsonl).
# A directory path auto-generates a filename inside it.
# Omit or comment out to disable.
# save_failures = \".\"

# Max retries per judgement on HTTP errors. 0 = no retries. Default: 3.
# retries = 3

# RNG seed for reproducible runs. Omit for OS entropy.
# seed = 42

# --- Judges ---
# At least one [[judge]] block is required. Each judge needs endpoint, model,
# and temperature. All other per-judge fields are optional.
#
# Per-judge fields:
#   endpoint (required)    - OpenAI-compatible API base URL
#   model (required)       - Model ID
#   temperature (required) - LLM sampling temperature
#   concurrency            - Max concurrent requests (default: global concurrency or 16)
#   weight                 - Relative weight for pair assignment (default: 1)
#   temperature_jitter     - Std dev of N(1.0, jitter) temperature multiplier (default: 0)
#   presence_penalty       - Penalizes repeated tokens, range -2.0 to 2.0
#   top_p                  - Nucleus sampling threshold, range 0.0 to 1.0
#   min_logprob_coverage   - Min verdict-token logprob mass to trust a verdict, > 0.0 and <= 1.0 (default: 0.95)
#   verdict_temperature    - Temper this judge's parsed verdicts, q^(1/T); > 1 softens overconfidence, must be > 0 (default: global value, else 3.0 reasoning / 1.0 no-reasoning)
#   max_tokens             - Maximum tokens in LLM response (default: 2048, or average of specified judges)
#   api_key_env            - Environment variable name containing the API key
#   reasoning_effort       - OpenRouter: controls reasoning/thinking mode (e.g. \\\"none\\\" to disable Qwen thinking)
#   chat_template_kwargs   - Extra kwargs passed to the server's chat template (e.g. enable_thinking = false for llama.cpp)

[[judge]]
endpoint = \"http://localhost:8000\"
model = \"my-model\"
temperature = 0.7
# concurrency = 16
# weight = 1
# max_tokens = 2048

# [[judge]]
# endpoint = \"https://api.openai.com/v1\"
# model = \"gpt-4o\"
# api_key_env = \"OPENAI_API_KEY\"
# concurrency = 5
# weight = 3
# temperature = 1.0

# --- Scoring hyperparameters ---
# Most users should not need to change these.

# Confidence interval level. Default: 0.95 (95%).
# confidence_level = 0.95

# Ghost player regularization strength. Default: 0.01.
# regularization_strength = 0.01

# Positional bias prior in probability space. 0.5 = no bias (default).
# Values > 0.5 mean the model tends to favor the first-listed item.
# Must be > 0.0 and < 1.0 (exclusive).
# bias_prior = 0.5

# Info-gain exponent for matchmaking. Higher = more exploitation. Default: 1.0.
# matchmaking_sharpness = 1.0

# Minimum edges per item before the top-heavy distribution kicks in.
# Must be at least 1. Default: 2.
# min_uniform_edges = 2

# Prior variance on log-strengths. Default: 10.0.
# prior_tau2 = 10.0

# Prior variance on positional bias (logit space). Default: 2.0.
# bias_prior_tau2 = 2.0
";

/// Returns the default config path.
/// Linux: ~/.config/nanojudge/config.toml
/// macOS: ~/Library/Application Support/nanojudge/config.toml
/// Windows: C:\Users\<user>\AppData\Roaming\nanojudge\config.toml
pub fn config_path() -> PathBuf {
    let config_dir = dirs::config_dir().unwrap_or_else(|| bail("Could not determine config directory"));
    config_dir.join("nanojudge").join("config.toml")
}

/// Load config from a file path. Returns default (all None) if file doesn't exist.
pub fn load_config(path: &Path) -> NanojudgeConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            toml::from_str(&content).unwrap_or_else(|e| {
                let mut msg = format!("Bad config at {}:\n  {e}", path.display());
                if let Some(span) = e.span() {
                    let line_num = content[..span.start].matches('\n').count() + 1;
                    if let Some(line) = content.lines().nth(line_num - 1) {
                        msg.push_str(&format!("\n\n  line {line_num}: {}", line.trim()));
                    }
                }
                bail(msg)
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => NanojudgeConfig::default(),
        Err(e) => bail(format!("Failed to read config at {}: {e}", path.display())),
    }
}

/// Create the default config file. Errors if it already exists.
pub fn create_default_config() -> PathBuf {
    let path = config_path();

    if path.exists() {
        bail(format!("Config file already exists at {}", path.display()));
    }

    // Create parent directories
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| bail(format!("Failed to create directory {}: {e}", parent.display())));
    }

    std::fs::write(&path, DEFAULT_CONFIG_TEMPLATE)
        .unwrap_or_else(|e| bail(format!("Failed to write config to {}: {e}", path.display())));

    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_config_path_ends_correctly() {
        let path = config_path();
        assert!(path.ends_with("nanojudge/config.toml"));
    }

    #[test]
    fn test_load_config_missing_file_returns_defaults() {
        let config = load_config(Path::new("/tmp/nanojudge-test-nonexistent/config.toml"));
        assert!(config.rounds.is_none());
        assert!(config.concurrency.is_none());
        assert!(config.judge.is_none());
    }

    #[test]
    fn test_load_config_parses_global_fields() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
rounds = 10
concurrency = 16
logprobs = false
analysis_length = "3 sentences"

[[judge]]
endpoint = "http://localhost:8000"
model = "qwen-4b"
temperature = 0.7
temperature_jitter = 0.05
presence_penalty = 1.5
top_p = 0.95
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.rounds.unwrap(), 10);
        assert_eq!(config.concurrency.unwrap(), 16);
        assert!(!config.logprobs.unwrap());
        assert_eq!(config.analysis_length.as_deref().unwrap(), "3 sentences");
        let judges = config.judge.unwrap();
        assert_eq!(judges.len(), 1);
        assert_eq!(judges[0].endpoint, "http://localhost:8000");
        assert_eq!(judges[0].model, "qwen-4b");
        assert_eq!(judges[0].temperature.unwrap(), 0.7);
        assert_eq!(judges[0].temperature_jitter.unwrap(), 0.05);
        assert_eq!(judges[0].presence_penalty.unwrap(), 1.5);
        assert_eq!(judges[0].top_p.unwrap(), 0.95);
    }

    #[test]
    fn test_load_config_partial_fields() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
rounds = 5
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.rounds.unwrap(), 5);
        assert!(config.judge.is_none());
        assert!(config.concurrency.is_none());
    }

    #[test]
    fn test_default_config_template_parses() {
        let config: NanojudgeConfig = toml::from_str(DEFAULT_CONFIG_TEMPLATE).unwrap();
        // Template has one uncommented [[judge]] block
        assert!(config.judge.is_some());
        assert_eq!(config.judge.as_ref().unwrap().len(), 1);
        assert!(config.rounds.is_none());
    }

    #[test]
    fn test_parse_judge_blocks() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
rounds = 10

[[judge]]
endpoint = "http://localhost:8000"
model = "qwen-32b"
concurrency = 50
weight = 5.0
temperature = 0.6

[[judge]]
endpoint = "https://api.openai.com/v1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
concurrency = 5
weight = 3.0
temperature = 1.0
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.rounds.unwrap(), 10);
        let judges = config.judge.unwrap();
        assert_eq!(judges.len(), 2);
        assert_eq!(judges[0].endpoint, "http://localhost:8000");
        assert_eq!(judges[0].model, "qwen-32b");
        assert_eq!(judges[0].concurrency.unwrap(), 50);
        assert_eq!(judges[0].weight.unwrap(), 5.0);
        assert_eq!(judges[0].temperature.unwrap(), 0.6);
        assert_eq!(judges[1].endpoint, "https://api.openai.com/v1");
        assert_eq!(judges[1].model, "gpt-4o");
        assert_eq!(judges[1].api_key_env.as_deref().unwrap(), "OPENAI_API_KEY");
        assert_eq!(judges[1].concurrency.unwrap(), 5);
    }

    #[test]
    fn test_no_judge_blocks_returns_none() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
rounds = 5
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert!(config.judge.is_none());
        assert_eq!(config.rounds.unwrap(), 5);
    }

    #[test]
    fn test_reasoning_enabled_parses() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
reasoning_enabled = false
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.reasoning_enabled, Some(false));
    }

    #[test]
    fn test_verdict_temperature_parses() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
verdict_temperature = 6.0

[[judge]]
endpoint = "http://localhost:8000"
model = "qwen-32b"
temperature = 0.6
verdict_temperature = 2.5
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert_eq!(config.verdict_temperature, Some(6.0));
        assert_eq!(config.judge.unwrap()[0].verdict_temperature, Some(2.5));
    }

    #[test]
    fn test_reasoning_enabled_defaults_to_none() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
rounds = 5
"#).unwrap();

        let config = load_config(tmpfile.path());
        assert!(config.reasoning_enabled.is_none());
    }

    #[test]
    fn test_unknown_top_level_field_rejected() {
        let input = "rounds = 10\nbogus_field = true\n";
        let result: Result<NanojudgeConfig, _> = toml::from_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("bogus_field"), "error should name the bad field: {err}");
    }

    #[test]
    fn test_unknown_judge_field_rejected() {
        let input = r#"
[[judge]]
endpoint = "http://localhost:8000"
model = "test"
judgement_distribution = "top-heavy"
"#;
        let result: Result<NanojudgeConfig, _> = toml::from_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("judgement_distribution"), "error should name the bad field: {err}");
    }

    #[test]
    fn test_misspelled_field_rejected() {
        let input = "roudns = 10\n";
        let result: Result<NanojudgeConfig, _> = toml::from_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("roudns"), "error should name the bad field: {err}");
    }
}
