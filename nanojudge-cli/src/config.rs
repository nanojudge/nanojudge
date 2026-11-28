/// Config file loading and creation for nanojudge CLI.
///
/// Config lives at ~/.config/nanojudge/config.toml.
/// All fields are optional — CLI args override config values.
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::args::OutputFormat;
use crate::Error;

/// Per-judge configuration from a [[judge]] TOML block.
#[derive(Deserialize, Default, Clone)]
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
    /// Bearer token supplied directly in memory (rejected in TOML files so
    /// config files cannot accidentally store plaintext keys). Takes precedence
    /// over `api_key_env` and any global fallback key.
    #[serde(skip)]
    pub api_key: Option<String>,
    pub max_tokens: Option<u32>,
    /// Sent as `reasoning_effort`: controls the model's reasoning.
    /// Set to "none" to turn reasoning off.
    pub reasoning_effort: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, toml::Value>>,
    /// OpenRouter provider routing options, sent as `provider`. Required
    /// for OpenRouter judges.
    pub provider: Option<HashMap<String, toml::Value>>,
    /// Seconds before a request to this judge is abandoned and retried. No
    /// default: a reasonable limit varies too much with the prompt, the model
    /// and the hardware the model runs on. Unset waits indefinitely.
    pub request_timeout_secs: Option<f64>,
}

impl std::fmt::Debug for JudgeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Exhaustive destructuring ensures any newly added field must be handled here.
        let Self {
            endpoint,
            model,
            concurrency,
            weight,
            temperature,
            temperature_jitter,
            presence_penalty,
            top_p,
            min_logprob_coverage,
            verdict_temperature,
            api_key_env,
            api_key,
            max_tokens,
            reasoning_effort,
            chat_template_kwargs,
            provider,
            request_timeout_secs,
        } = self;
        f.debug_struct("JudgeConfig")
            .field("endpoint", endpoint)
            .field("model", model)
            .field("concurrency", concurrency)
            .field("weight", weight)
            .field("temperature", temperature)
            .field("temperature_jitter", temperature_jitter)
            .field("presence_penalty", presence_penalty)
            .field("top_p", top_p)
            .field("min_logprob_coverage", min_logprob_coverage)
            .field("verdict_temperature", verdict_temperature)
            .field("api_key_env", api_key_env)
            .field("api_key", &api_key.as_ref().map(|_| "[REDACTED]"))
            .field("max_tokens", max_tokens)
            .field("reasoning_effort", reasoning_effort)
            .field("chat_template_kwargs", chat_template_kwargs)
            .field("provider", provider)
            .field("request_timeout_secs", request_timeout_secs)
            .finish()
    }
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct NanojudgeConfig {
    pub judgements_per_item: Option<usize>,
    pub concurrency: Option<usize>,
    pub prompt_template: Option<String>,
    pub logprobs: Option<bool>,
    pub deliberation_length: Option<String>,
    pub judgement_distribution: Option<String>,
    /// Number of items in each judged lineup: 2 to 9.
    pub lineup_size: Option<usize>,
    pub selection_sharpness: Option<f64>,
    pub selection_top_k: Option<usize>,
    pub selection_draws: Option<usize>,
    pub cutoff: Option<f64>,
    pub coverage: Option<f64>,
    pub stop_confidence: Option<f64>,
    pub retries: Option<usize>,
    pub confidence_level: Option<f64>,
    pub regularization_strength: Option<f64>,
    pub bias_prior: Option<f64>,
    pub matchmaking_sharpness: Option<f64>,
    pub min_uniform_edges: Option<usize>,
    pub judgements_per_refit: Option<usize>,
    pub prior_tau2: Option<f64>,
    pub bias_prior_tau2: Option<f64>,
    pub deliberation_enabled: Option<bool>,
    pub min_logprob_coverage: Option<f64>,
    pub verdict_temperature: Option<f64>,
    pub live_top: Option<usize>,
    pub save_successful_judgements: Option<PathBuf>,
    pub save_prompt_text: Option<bool>,
    pub save_response_text: Option<bool>,
    pub save_reasoning_text: Option<bool>,
    pub output_format: Option<OutputFormat>,
    pub verbose: Option<bool>,
    pub save_failed_judgements: Option<PathBuf>,
    pub seed: Option<u64>,
    /// Judge panel configuration. At least one [[judge]] block is required.
    pub judge: Option<Vec<JudgeConfig>>,
}

const DEFAULT_CONFIG_TEMPLATE: &str = "\
# nanojudge configuration
# All values here can be overridden by CLI flags unless noted otherwise.

# Average judgements per item. Total budget = ceil(judgements_per_item * num_items / lineup_size).
# judgements_per_item = 10

# Default max concurrent LLM requests per judge (can be overridden per-judge)
# concurrency = 16

# Path to a custom prompt template file.
# Pairwise templates must contain $criterion, $option1 and $option2
# ($name1, $name2 and $length are optional). Lineup templates must contain
# $criterion and $optionA through the letter for the lineup size.
# If not set, the built-in default prompt is used.
# prompt_template = \"/path/to/my-prompt.txt\"

# Enable logprob extraction for continuous win probabilities.
# Requires an endpoint that supports logprobs (e.g. vLLM, OpenAI direct).
# When false, uses text-based verdict parsing (discrete) — may need more judgements.
# logprobs = false

# How much deliberation the LLM should write before its verdict.
# Examples: \"3 sentences\", \"1 paragraph\", \"5 sentences\".
# deliberation_length = \"2 paragraphs\"

# Whether to have the LLM deliberate before giving its verdict.
# When false, the LLM skips deliberation and outputs only the verdict.
# Faster and cheaper, but may reduce accuracy. Judges without max_tokens
# default to just enough for the verdict (16 for pairs, one ranking line per
# item for lineups).
# With a custom prompt_template, set this to match what the template asks for:
# false means the template must ask for the verdict only.
# deliberation_enabled = true

# Judgement distribution: \"uniform\" or \"top-heavy\".
# Uniform gives equal attention to all items. Top-heavy focuses on contenders.
# judgement_distribution = \"top-heavy\"

# Number of items in each judged lineup: 2 (default) up to 9.
# Above 2, each judgement ranks the whole lineup and is folded into one edge per
# pair, so one LLM call yields more information. Requires logprobs for the
# full benefit. A custom prompt_template must name exactly as many options as
# the lineup size ($optionA through $optionI).
# lineup_size = 2

# Tuning for the top-heavy distribution. Each item's pairing weight is its
# uncertainty ratio: min/max of its chance of being in vs out of the top k
# (`selection_top_k`) — 1 when the item is a coin flip for the top k, near 0
# once it is confidently in or out — raised to `selection_sharpness`. The
# first item of each judgement is drawn from these weights; its opponent is
# picked by info-gain matchmaking from a rating window around it, so the
# contested boundary gets the judgements.
#   selection_sharpness: lower = flatter = more exploration; 1.0 = raw ratios (> 0).
#   selection_top_k: how many top places to settle. 1 = first place; 10 is
#             right for \"find the top ten, order within them doesn't
#             matter\". Must be >= 1 and <= number of items.
#   selection_draws: random draws used at every refit to estimate each item's
#             chance of being in the top k. More = steadier estimate, more
#             computation. Must be >= 1.
#   cutoff:   minimum uncertainty ratio to stay a candidate; items below are
#             dropped, except the two highest, which are always kept.
#             [0,1); 0 disables the cutoff.
#   coverage: proportional-fair pull. Each weight is divided by
#             edge-count^coverage, pulling cumulative edges toward the
#             ratio-implied share. 0 = off, 1 = proportional-fair, > 1 over-corrects.
# Only used with judgement_distribution = \"top-heavy\".
# selection_sharpness = 0.7
# selection_top_k = 1
# selection_draws = 1000
# cutoff = 0.0
# coverage = 1.0

# Early stop for top-heavy runs. After each interim fit, the run ends early
# once the probability that every item is on its likelier side of the top-k
# boundary (in or out) — the product of the per-item side probabilities —
# reaches this value; final
# scoring then runs on the judgements collected so far. In (0.5, 1.0), e.g.
# 0.95. No default: when unset, the run always uses its full budget.
# Rejected with judgement_distribution = \"uniform\".
# stop_confidence = 0.95

# Minimum fraction of verdict-token probability mass the top-logprobs must
# cover for a logprob-based verdict to be trusted (logprobs mode only) — the
# option digits for pairwise, the option letters for a lineup. Below this, the
# judgement is treated as unparseable. Must be > 0.0 and <= 1.0.
# Can be overridden per-judge.
# min_logprob_coverage = 0.95

# Temperature applied to each parsed verdict distribution before scoring:
# q^(1/T), which divides every edge's log-odds by T. 1.0 = off. > 1 pulls
# overconfident verdicts toward 50/50; < 1 sharpens. Distinct from the
# sampling `temperature` — this tempers the parsed verdict, not the LLM call.
# Text-mode (non-logprob) verdicts are one-hot and unaffected.
# Must be finite and > 0. Can be overridden per-judge.
# Default: 3.0 when deliberation is enabled (a verdict token written after
# deliberating is near-deterministic, so its logprobs read overconfident);
# 1.0 when deliberation is disabled.
# verdict_temperature = 3.0

# Print a live ranking table after each refit. 0 = all items, N = top N.
# Omit or comment out to disable.
# live_top = 10

# Save successful judgements to a JSONL file. Value is a path.
# Use \".\" for the current directory (auto-generates judgements-{timestamp}.jsonl).
# A directory path auto-generates a filename inside it.
# By default saves metadata and verdicts but no text; see the three
# save_*_text settings below.
# Omit or comment out to disable.
# save_successful_judgements = \".\"

# Include the prompt sent to the LLM in saved successful judgements.
# Requires save_successful_judgements.
# save_prompt_text = false

# Include the LLM's visible answer (deliberation and verdict) in saved
# successful judgements. Requires save_successful_judgements.
# save_response_text = false

# Include the model's own reasoning text in saved successful judgements.
# Requires save_successful_judgements.
# save_reasoning_text = false

# Output format for final results: \"table\" or \"json\".
# output_format = \"table\"

# Show progress during execution.
# verbose = false

# Save failed judgements (unparseable responses) to a JSONL file. Value is a path.
# Use \".\" for the current directory (auto-generates failures-{timestamp}.jsonl).
# A directory path auto-generates a filename inside it.
# Always includes the prompt, response and reasoning text for debugging.
# Omit or comment out to disable.
# save_failed_judgements = \".\"

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
#   top_p                  - Nucleus sampling threshold, > 0.0 and <= 1.0
#   min_logprob_coverage   - Min verdict-token logprob mass to trust a verdict, > 0.0 and <= 1.0 (default: 0.95)
#   verdict_temperature    - Temper this judge's parsed verdicts, q^(1/T); > 1 softens overconfidence, must be > 0 (default: global value, else 3.0 deliberation / 1.0 no-deliberation)
#   max_tokens             - Maximum tokens in LLM response, required unless reasoning_effort = \"none\" (then default: 2048, or average of specified judges; without deliberation, just enough for the verdict)
#   api_key_env            - Environment variable name containing the API key
#   reasoning_effort       - Controls the model's reasoning (e.g. \"none\" to turn it off)
#   chat_template_kwargs   - Extra kwargs passed to the server's chat template (e.g. enable_thinking = false for llama.cpp)
#   provider               - OpenRouter provider routing options, required for OpenRouter judges (e.g. { only = [\"xiaomi\"], allow_fallbacks = false })
#   request_timeout_secs   - Seconds before a request is abandoned and retried, > 0 (e.g. 300 or 2.5). No default, as a
#                            reasonable limit depends heavily on the prompt, the model and its hardware; unset waits indefinitely

[[judge]]
endpoint = \"http://localhost:8000\"
model = \"my-model\"
temperature = 0.7
max_tokens = 2048
# concurrency = 16
# weight = 1

# [[judge]]
# endpoint = \"https://api.openai.com/v1\"
# model = \"gpt-4o\"
# api_key_env = \"OPENAI_API_KEY\"
# concurrency = 5
# weight = 3
# temperature = 1.0
# max_tokens = 2048

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

# Number of judgement attempts scheduled between refits. The value is literal,
# including during uniform pairing. Default: the number needed for every item
# to appear at least once.
# judgements_per_refit = 50

# Prior variance on log-strengths. Default: 10.0.
# prior_tau2 = 10.0

# Prior variance on positional bias (logit space). Default: 2.0.
# bias_prior_tau2 = 2.0
";

/// Returns the default config path.
/// Linux: ~/.config/nanojudge/config.toml
/// macOS: ~/Library/Application Support/nanojudge/config.toml
/// Windows: C:\Users\<user>\AppData\Roaming\nanojudge\config.toml
pub fn config_path() -> crate::Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| Error::Io("Could not determine config directory".into()))?;
    Ok(config_dir.join("nanojudge").join("config.toml"))
}

/// Resolve the config path (`--config` if given, otherwise the default path)
/// and load it. Errors if the file is missing, unreadable, or invalid TOML.
pub fn load_config(explicit_path: Option<&Path>) -> crate::Result<(PathBuf, NanojudgeConfig)> {
    let (path, is_default) = match explicit_path {
        Some(p) => (p.to_path_buf(), false),
        None => (config_path()?, true),
    };
    let cfg = try_load_config(&path, is_default)?;
    Ok((path, cfg))
}

/// Parse a [`NanojudgeConfig`] from a TOML string.
pub fn parse_config(content: &str) -> crate::Result<NanojudgeConfig> {
    parse_config_with_source(content, None)
}

fn parse_config_with_source(content: &str, path: Option<&Path>) -> crate::Result<NanojudgeConfig> {
    toml::from_str(content).map_err(|e| {
        let mut msg = match path {
            Some(p) => format!("Bad config at {}:\n  {e}", p.display()),
            None => format!("Bad config:\n  {e}"),
        };
        if let Some(span) = e.span() {
            let line_num = content[..span.start].matches('\n').count() + 1;
            if let Some(line) = content.lines().nth(line_num - 1) {
                msg.push_str(&format!("\n\n  line {line_num}: {}", line.trim()));
            }
        }
        Error::Config(msg)
    })
}

fn try_load_config(path: &Path, is_default: bool) -> crate::Result<NanojudgeConfig> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::Config(if is_default {
                format!(
                    "Config file not found at {}. Run `nanojudge init` to create one.",
                    path.display()
                )
            } else {
                format!("Config file not found: {}", path.display())
            }));
        }
        Err(e) => return Err(Error::Io(format!("Failed to read config at {}: {e}", path.display()))),
    };
    parse_config_with_source(&content, Some(path))
}

/// Create the default config file. Errors if it already exists.
pub fn create_default_config() -> crate::Result<PathBuf> {
    let path = config_path()?;

    if path.exists() {
        return Err(Error::Config(format!("Config file already exists at {}", path.display())));
    }

    // Create parent directories
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Io(format!("Failed to create directory {}: {e}", parent.display())))?;
    }

    std::fs::write(&path, DEFAULT_CONFIG_TEMPLATE)
        .map_err(|e| Error::Io(format!("Failed to write config to {}: {e}", path.display())))?;

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_config_path_ends_correctly() {
        let path = config_path().unwrap();
        assert!(path.ends_with("nanojudge/config.toml"));
    }

    #[test]
    fn test_load_config_missing_file_errors() {
        let missing = Path::new("/tmp/nanojudge-test-nonexistent/config.toml");

        let err_explicit = try_load_config(missing, false).unwrap_err();
        assert!(matches!(err_explicit, Error::Config(_)), "{err_explicit:?}");
        let msg_explicit = err_explicit.to_string();
        assert!(msg_explicit.contains("Config file not found"), "{msg_explicit}");
        assert!(msg_explicit.contains("config.toml"), "{msg_explicit}");
        assert!(!msg_explicit.contains("nanojudge init"), "{msg_explicit}");

        let err_default = try_load_config(missing, true).unwrap_err();
        assert!(matches!(err_default, Error::Config(_)), "{err_default:?}");
        let msg_default = err_default.to_string();
        assert!(msg_default.contains("Config file not found"), "{msg_default}");
        assert!(msg_default.contains("nanojudge init"), "{msg_default}");
    }

    #[test]
    fn test_load_config_parses_global_fields() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
judgements_per_item = 10
judgements_per_refit = 3
concurrency = 16
logprobs = false
deliberation_length = "3 sentences"

[[judge]]
endpoint = "http://localhost:8000"
model = "qwen-4b"
temperature = 0.7
temperature_jitter = 0.05
presence_penalty = 1.5
top_p = 0.95
"#).unwrap();

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert_eq!(config.judgements_per_item.unwrap(), 10);
        assert_eq!(config.judgements_per_refit, Some(3));
        assert_eq!(config.concurrency.unwrap(), 16);
        assert!(!config.logprobs.unwrap());
        assert_eq!(config.deliberation_length.as_deref().unwrap(), "3 sentences");
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
judgements_per_item = 5
"#).unwrap();

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert_eq!(config.judgements_per_item.unwrap(), 5);
        assert!(config.judge.is_none());
        assert!(config.concurrency.is_none());
    }

    #[test]
    fn test_default_config_template_parses() {
        let config: NanojudgeConfig = toml::from_str(DEFAULT_CONFIG_TEMPLATE).unwrap();
        // Template has one uncommented [[judge]] block
        assert!(config.judge.is_some());
        assert_eq!(config.judge.as_ref().unwrap().len(), 1);
        assert!(config.judgements_per_item.is_none());
    }

    #[test]
    fn test_parse_judge_blocks() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
judgements_per_item = 10

[[judge]]
endpoint = "http://localhost:8000"
model = "qwen-32b"
concurrency = 50
weight = 5.0
temperature = 0.6
request_timeout_secs = 300

[[judge]]
endpoint = "https://api.openai.com/v1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
concurrency = 5
weight = 3.0
temperature = 1.0
"#).unwrap();

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert_eq!(config.judgements_per_item.unwrap(), 10);
        let judges = config.judge.unwrap();
        assert_eq!(judges.len(), 2);
        assert_eq!(judges[0].endpoint, "http://localhost:8000");
        assert_eq!(judges[0].model, "qwen-32b");
        assert_eq!(judges[0].concurrency.unwrap(), 50);
        assert_eq!(judges[0].weight.unwrap(), 5.0);
        assert_eq!(judges[0].temperature.unwrap(), 0.6);
        assert_eq!(judges[0].request_timeout_secs, Some(300.0));
        assert_eq!(judges[1].request_timeout_secs, None);
        assert_eq!(judges[1].endpoint, "https://api.openai.com/v1");
        assert_eq!(judges[1].model, "gpt-4o");
        assert_eq!(judges[1].api_key_env.as_deref().unwrap(), "OPENAI_API_KEY");
        assert_eq!(judges[1].concurrency.unwrap(), 5);
    }

    #[test]
    fn test_no_judge_blocks_returns_none() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
judgements_per_item = 5
"#).unwrap();

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert!(config.judge.is_none());
        assert_eq!(config.judgements_per_item.unwrap(), 5);
    }

    #[test]
    fn test_deliberation_enabled_parses() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
deliberation_enabled = false
"#).unwrap();

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert_eq!(config.deliberation_enabled, Some(false));
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

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert_eq!(config.verdict_temperature, Some(6.0));
        assert_eq!(config.judge.unwrap()[0].verdict_temperature, Some(2.5));
    }

    #[test]
    fn test_deliberation_enabled_defaults_to_none() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        write!(tmpfile, r#"
judgements_per_item = 5
"#).unwrap();

        let config = load_config(Some(tmpfile.path())).unwrap().1;
        assert!(config.deliberation_enabled.is_none());
    }

    #[test]
    fn test_unknown_top_level_field_rejected() {
        let input = "judgements_per_item = 10\nbogus_field = true\n";
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

    #[test]
    fn test_parse_config_and_reject_plaintext_api_key_in_toml() {
        let cfg = parse_config("judgements_per_item = 4\n").unwrap();
        assert_eq!(cfg.judgements_per_item, Some(4));

        // In-memory api_key must stay rejected when written into TOML.
        let err = parse_config(
            "[[judge]]\nendpoint = \"http://localhost:8000\"\nmodel = \"m\"\napi_key = \"sk-secret\"\n",
        )
        .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
        assert!(err.to_string().contains("api_key"), "{err}");

        // Debug output must not expose an in-memory api_key.
        let judge = JudgeConfig {
            endpoint: "http://localhost:8000".into(),
            model: "m".into(),
            api_key: Some("sk-super-secret".into()),
            ..Default::default()
        };
        let dbg = format!("{judge:?}");
        assert!(!dbg.contains("sk-super-secret"), "{dbg}");
        assert!(dbg.contains("[REDACTED]"), "{dbg}");
    }
}
