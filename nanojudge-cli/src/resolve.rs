use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use nanojudge_core::constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE};
use nanojudge_core::{JudgementDistribution, judge_hash};

use crate::args::{ConfigArgs, OutputFormat};
use crate::bail;
use crate::config;
use crate::parse;
use crate::prompt;
use crate::{
    DEFAULT_BIAS_PRIOR, DEFAULT_BIAS_PRIOR_TAU2, DEFAULT_CONFIDENCE_LEVEL, DEFAULT_PRIOR_TAU2,
    DEFAULT_REGULARIZATION_STRENGTH,
};

const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_TEMPERATURE_JITTER: f64 = 0.0;
const DEFAULT_MAX_RETRIES: usize = 3;
const DEFAULT_ANALYSIS_LENGTH: &str = "2 paragraphs";
const DEFAULT_TARGET_PRIOR_EDGES: f64 = 5.0;
// A verdict token written after a reasoning analysis is near-deterministic, so
// its logprobs read overconfident and get decompressed by default. Without
// reasoning, the verdict token is the model's first expression of preference
// and its logprobs are left untouched.
pub(crate) const DEFAULT_VERDICT_TEMPERATURE_REASONING: f64 = 3.0;
pub(crate) const DEFAULT_VERDICT_TEMPERATURE_NO_REASONING: f64 = 1.0;

/// Merge a CLI value with a config file value. CLI wins.
/// Warns to stderr if both are set and differ.
fn merge_opt<T: PartialEq + std::fmt::Display>(
    cli: Option<T>,
    cfg: Option<T>,
    flag: &str,
) -> Option<T> {
    match (cli, cfg) {
        (Some(c), Some(f)) => {
            if c != f {
                eprintln!("Warning: --{flag} ({c}) overrides config file value ({f})");
            }
            Some(c)
        }
        (c @ Some(_), None) => c,
        (None, f) => f,
    }
}

/// Resolved configuration — CLI args merged with config file values.
/// All required values are concrete (no Options except genuinely optional ones).
pub struct ResolvedConfig {
    pub judgements_per_item: usize,
    pub judgement_distribution: JudgementDistribution,
    /// Number of items in each judged lineup: 2 (default) up to 9.
    pub lineup_size: usize,
    /// Top-heavy selection sharpness (power applied to each item's uncertainty
    /// ratio around the anchor). Finite and > 0.
    pub selection_sharpness: f64,
    /// Top-heavy anchor rank (0-based, fractional interpolates between adjacent
    /// ranks; 0 = the current leader). Finite and >= 0; the upper bound
    /// (num_items - 1) is enforced at rank time once the item count is known.
    pub anchor_index: f64,
    /// Top-heavy selection cutoff (minimum uncertainty ratio to stay a
    /// candidate). In [0, 1); 0 disables the cutoff.
    pub selection_cutoff: f64,
    /// Top-heavy proportional-fair coverage pull. Finite and >= 0; 0 disables it.
    pub selection_coverage: f64,
    /// Top-heavy target blend: prior-predicted anchor counts as this many pseudo-edges.
    /// Finite and >= 0; 0 disables the blend.
    pub target_prior_edges: f64,
    /// Early-stop confidence for top-heavy runs: end the run once the
    /// probability that every item sits on its side of the anchor (the
    /// product of the per-item side probabilities) reaches this value. In
    /// (0.5, 1.0) when set. `None` = no early stop (the run always uses its
    /// full budget) — there is deliberately no default.
    pub stop_confidence: Option<f64>,
    pub retries: usize,
    pub analysis_length: String,
    pub reasoning_enabled: bool,
    pub prompt_template: String,
    pub confidence_level: f64,
    pub regularization_strength: f64,
    pub bias_prior_logit: f64,
    pub matchmaking_sharpness: f64,
    pub min_uniform_edges: usize,
    pub judgements_per_refit: Option<usize>,
    pub prior_tau2: f64,
    pub bias_prior_tau2: f64,
    pub live_top: Option<usize>,
    pub emit_interim_rankings: bool,
    pub save_successful_judgements: Option<PathBuf>,
    pub include_successful_prompts: bool,
    pub output_format: OutputFormat,
    pub verbose: bool,
    pub save_failed_judgements: Option<PathBuf>,
    pub seed: Option<u64>,
}

/// A resolved judge — all fields concrete, ready to build LlmConfig.
pub struct ResolvedJudge {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: f64,
    pub temperature_jitter: f64,
    pub presence_penalty: Option<f64>,
    pub top_p: Option<f64>,
    pub logprobs: bool,
    pub concurrency: usize,
    pub weight: f64,
    pub min_logprob_coverage: f64,
    /// Temperature applied to this judge's parsed verdict distributions
    /// (q^(1/T)) before edges are built. Finite and > 0; 1.0 = identity.
    pub verdict_temperature: f64,
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
    pub judge_id: u64,
    pub display_name: String,
}

fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::json!(i),
        toml::Value::Float(f) => serde_json::json!(f),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Array(a) => serde_json::Value::Array(a.iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => {
            let m = t.iter().map(|(k, v)| (k.clone(), toml_to_json(v))).collect();
            serde_json::Value::Object(m)
        }
        toml::Value::Datetime(d) => serde_json::Value::String(d.to_string()),
    }
}

/// Resolve judges from [[judge]] blocks in the config file.
/// Errors if no [[judge]] blocks are defined.
pub fn resolve_judges(
    shared: &ConfigArgs,
    cfg: &config::NanojudgeConfig,
    config_path: &Path,
    reasoning_enabled: bool,
) -> Vec<ResolvedJudge> {
    let judge_configs = cfg.judge.as_ref().filter(|j| !j.is_empty())
        .unwrap_or_else(|| {
            bail(format!(
                "No [[judge]] blocks defined in {}. At least one judge is required.\n\
                 Example:\n\n  [[judge]]\n  endpoint = \"http://localhost:8000\"\n  model = \"my-model\"\n  temperature = 0.7",
                config_path.display()
            ));
        });

    // Validate: no duplicate endpoint+model
    let mut seen = HashSet::new();
    for jc in judge_configs {
        let key = format!("{}\0{}", jc.endpoint, jc.model);
        if !seen.insert(key) {
            bail(format!(
                "Duplicate judge: endpoint=\"{}\" model=\"{}\". Each judge must have a unique endpoint+model combination.",
                jc.endpoint, jc.model
            ));
        }
    }

    let global_logprobs = merge_opt(shared.logprobs, cfg.logprobs, "logprobs")
        .unwrap_or(false);

    let global_concurrency = merge_opt(shared.concurrency, cfg.concurrency, "concurrency")
        .unwrap_or(DEFAULT_CONCURRENCY);
    if global_concurrency == 0 {
        bail("concurrency must be at least 1");
    }

    let global_min_logprob_coverage = merge_opt(shared.min_logprob_coverage, cfg.min_logprob_coverage, "min-logprob-coverage")
        .unwrap_or(parse::DEFAULT_MIN_LOGPROB_COVERAGE);
    if !global_min_logprob_coverage.is_finite() || global_min_logprob_coverage <= 0.0 || global_min_logprob_coverage > 1.0 {
        bail(format!(
            "min_logprob_coverage={global_min_logprob_coverage}, must be finite, > 0.0 and <= 1.0"
        ));
    }

    let global_verdict_temperature = merge_opt(shared.verdict_temperature, cfg.verdict_temperature, "verdict-temperature")
        .unwrap_or(if reasoning_enabled {
            DEFAULT_VERDICT_TEMPERATURE_REASONING
        } else {
            DEFAULT_VERDICT_TEMPERATURE_NO_REASONING
        });
    if !global_verdict_temperature.is_finite() || global_verdict_temperature <= 0.0 {
        bail(format!(
            "verdict_temperature={global_verdict_temperature}, must be finite and > 0"
        ));
    }

    // CLI --api-key or OPENAI_API_KEY env var
    let cli_api_key = shared.api_key.clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    // Resolve max_tokens: per-judge → average of specified judges → 2048
    let specified_max_tokens: Vec<u32> = judge_configs.iter()
        .filter_map(|jc| jc.max_tokens)
        .collect();
    let default_max_tokens = if specified_max_tokens.is_empty() {
        2048
    } else {
        let sum: u32 = specified_max_tokens.iter().sum();
        sum / specified_max_tokens.len() as u32
    };
    // Print message if some judges are missing max_tokens and we're using the average
    if !specified_max_tokens.is_empty() && specified_max_tokens.len() < judge_configs.len() {
        let missing: Vec<&str> = judge_configs.iter()
            .filter(|jc| jc.max_tokens.is_none())
            .map(|jc| jc.model.as_str())
            .collect();
        let specified: Vec<String> = judge_configs.iter()
            .filter_map(|jc| jc.max_tokens.map(|_| jc.model.clone()))
            .collect();
        eprintln!(
            "max_tokens not set for {}; using {} (average of {})",
            missing.join(", "), default_max_tokens, specified.join(", ")
        );
    }

    // Compute display names: model is default, disambiguate with the endpoint if models collide
    let mut model_counts: HashMap<String, usize> = HashMap::new();
    for jc in judge_configs {
        *model_counts.entry(jc.model.clone()).or_insert(0) += 1;
    }

    let mut judges: Vec<ResolvedJudge> = Vec::with_capacity(judge_configs.len());

    for jc in judge_configs {
        let temperature = jc.temperature
            .unwrap_or_else(|| {
                bail(format!(
                    "No temperature specified for judge {}. Set temperature in the [[judge]] block.",
                    jc.model,
                ));
            });

        let api_key = if let Some(ref env_name) = jc.api_key_env {
            match std::env::var(env_name) {
                Ok(key) => Some(key),
                Err(_) => bail(format!(
                    "Judge {}: api_key_env = \"{}\" but that environment variable is not set.",
                    jc.model, env_name
                )),
            }
        } else {
            cli_api_key.clone()
        };

        let display_name = if model_counts[&jc.model] > 1 {
            // endpoint+model is unique (validated above), so the full endpoint
            // always disambiguates. The host alone does not when two judges
            // share a host and differ only by path.
            let endpoint = jc.endpoint
                .trim_start_matches("http://")
                .trim_start_matches("https://");
            format!("{} ({})", jc.model, endpoint)
        } else {
            jc.model.clone()
        };

        let judge_id = judge_hash(&jc.endpoint, &jc.model);
        let weight = jc.weight.unwrap_or(1.0);
        if !weight.is_finite() || weight < 0.0 {
            bail(format!("Judge {} has invalid weight {}. Weights must be finite and >= 0.", jc.model, weight));
        }

        let min_logprob_coverage = jc.min_logprob_coverage.unwrap_or(global_min_logprob_coverage);
        if !min_logprob_coverage.is_finite() || min_logprob_coverage <= 0.0 || min_logprob_coverage > 1.0 {
            bail(format!(
                "Judge {} has min_logprob_coverage={}, must be finite, > 0.0 and <= 1.0",
                jc.model, min_logprob_coverage
            ));
        }

        let verdict_temperature = jc.verdict_temperature.unwrap_or(global_verdict_temperature);
        if !verdict_temperature.is_finite() || verdict_temperature <= 0.0 {
            bail(format!(
                "Judge {} has verdict_temperature={}, must be finite and > 0",
                jc.model, verdict_temperature
            ));
        }

        judges.push(ResolvedJudge {
            endpoint: jc.endpoint.clone(),
            model: jc.model.clone(),
            api_key,
            temperature,
            temperature_jitter: jc.temperature_jitter.unwrap_or(DEFAULT_TEMPERATURE_JITTER),
            presence_penalty: jc.presence_penalty,
            top_p: jc.top_p,
            logprobs: global_logprobs,
            concurrency: {
                let c = jc.concurrency.unwrap_or(global_concurrency);
                if c == 0 {
                    bail(format!("judge '{}': concurrency must be at least 1", jc.endpoint));
                }
                c
            },
            weight,
            min_logprob_coverage,
            verdict_temperature,
            max_tokens: jc.max_tokens.unwrap_or(default_max_tokens),
            reasoning_effort: jc.reasoning_effort.clone(),
            chat_template_kwargs: jc.chat_template_kwargs.as_ref().map(|m| {
                m.iter().map(|(k, v)| (k.clone(), toml_to_json(v))).collect()
            }),
            judge_id,
            display_name,
        });
    }

    if judges.iter().map(|j| j.weight).sum::<f64>() <= 0.0 {
        bail("At least one judge must have positive weight.".to_string());
    }

    if !reasoning_enabled {
        let any_explicit_max_tokens = judge_configs.iter().any(|jc| jc.max_tokens.is_some());
        if any_explicit_max_tokens {
            eprintln!("Warning: max_tokens is ignored when reasoning is disabled (forced to 16)");
        }
        for j in &mut judges {
            j.max_tokens = 16;
            j.temperature = 0.0;
        }
    }

    judges
}

/// Resolve CLI args + config file + defaults into final config.
/// Judge-specific settings (endpoint, model, temperature, etc.) are handled by resolve_judges().
pub fn resolve_config(shared: &ConfigArgs, cfg: &config::NanojudgeConfig) -> ResolvedConfig {
    let judgements_per_item = merge_opt(shared.judgements_per_item, cfg.judgements_per_item, "judgements-per-item")
        .unwrap_or_else(|| {
            bail("--judgements-per-item is required (set it on the CLI or in the config file)");
        });
    if judgements_per_item == 0 {
        bail("--judgements-per-item must be at least 1");
    }

    let judgement_distribution_str = merge_opt(shared.judgement_distribution.clone(), cfg.judgement_distribution.clone(), "judgement-distribution")
        .unwrap_or_else(|| "top-heavy".to_string());
    let judgement_distribution = match judgement_distribution_str.as_str() {
        "uniform" => JudgementDistribution::Uniform,
        "top-heavy" => JudgementDistribution::TopHeavy,
        other => bail(format!("Unknown judgement distribution \"{other}\". Use \"uniform\" or \"top-heavy\".")),
    };

    let lineup_size = merge_opt(shared.lineup_size, cfg.lineup_size, "lineup-size")
        .unwrap_or(2);
    if !(MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE).contains(&lineup_size) {
        bail(format!(
            "lineup-size={lineup_size}, must be between {MIN_LINEUP_SIZE} and {MAX_LINEUP_SIZE}"
        ));
    }
    // Size 2 keeps the dedicated pairwise path (graded clear/narrow verdicts);
    // 3 and up run the lineup path.
    let uses_lineups = lineup_size >= 3;

    // Top-heavy selection tuning (only used with the top-heavy distribution).
    let selection_sharpness = merge_opt(shared.selection_sharpness, cfg.selection_sharpness, "selection-sharpness")
        .unwrap_or(0.7);
    if !selection_sharpness.is_finite() || selection_sharpness <= 0.0 {
        bail(format!("selection-sharpness={selection_sharpness}, must be finite and > 0"));
    }
    let anchor_index = merge_opt(shared.anchor_index, cfg.anchor_index, "anchor-index")
        .unwrap_or(0.0);
    if !anchor_index.is_finite() || anchor_index < 0.0 {
        bail(format!("anchor-index={anchor_index}, must be finite and >= 0 (0 anchors on the leader)"));
    }
    let selection_cutoff = merge_opt(shared.cutoff, cfg.cutoff, "cutoff")
        .unwrap_or(0.0);
    if !selection_cutoff.is_finite() || !(0.0..1.0).contains(&selection_cutoff) {
        bail(format!("cutoff={selection_cutoff}, must be in [0.0, 1.0) — 0 disables the cutoff"));
    }
    let selection_coverage = merge_opt(shared.coverage, cfg.coverage, "coverage")
        .unwrap_or(1.0);
    if !selection_coverage.is_finite() || selection_coverage < 0.0 {
        bail(format!("coverage={selection_coverage}, must be finite and >= 0 (0 disables it)"));
    }
    let target_prior_edges = merge_opt(shared.target_prior_edges, cfg.target_prior_edges, "target-prior-edges")
        .unwrap_or(DEFAULT_TARGET_PRIOR_EDGES);
    if !target_prior_edges.is_finite() || target_prior_edges < 0.0 {
        bail(format!("target-prior-edges={target_prior_edges}, must be finite and >= 0 (0 disables the blend)"));
    }
    // Early stop has deliberately no default: absent means the run always uses
    // its full budget.
    let stop_confidence = merge_opt(shared.stop_confidence, cfg.stop_confidence, "stop-confidence");
    if let Some(c) = stop_confidence {
        if !c.is_finite() || c <= 0.5 || c >= 1.0 {
            bail(format!("stop-confidence={c}, must be in (0.5, 1.0), e.g. 0.95"));
        }
        if matches!(judgement_distribution, JudgementDistribution::Uniform) {
            bail("stop-confidence requires judgement-distribution = \"top-heavy\" (uniform runs have no anchor to measure against)");
        }
    }
    let retries = merge_opt(shared.retries, cfg.retries, "retries")
        .unwrap_or(DEFAULT_MAX_RETRIES);
    let analysis_length = merge_opt(shared.analysis_length.clone(), cfg.analysis_length.clone(), "analysis-length")
        .unwrap_or_else(|| DEFAULT_ANALYSIS_LENGTH.to_string());

    let confidence_level = merge_opt(shared.confidence_level, cfg.confidence_level, "confidence-level")
        .unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
    if !confidence_level.is_finite() || confidence_level <= 0.0 || confidence_level >= 1.0 {
        bail(format!(
            "confidence-level={confidence_level}, must be between 0.0 and 1.0 (exclusive)"
        ));
    }
    let regularization_strength = merge_opt(shared.regularization_strength, cfg.regularization_strength, "regularization-strength")
        .unwrap_or(DEFAULT_REGULARIZATION_STRENGTH);
    if !regularization_strength.is_finite() || regularization_strength <= 0.0 {
        bail(format!("regularization-strength={regularization_strength}, must be finite and > 0"));
    }
    let matchmaking_sharpness = merge_opt(shared.matchmaking_sharpness, cfg.matchmaking_sharpness, "matchmaking-sharpness")
        .unwrap_or(1.0);
    if !matchmaking_sharpness.is_finite() || matchmaking_sharpness <= 0.0 {
        bail(format!("matchmaking-sharpness={matchmaking_sharpness}, must be finite and > 0"));
    }
    let min_uniform_edges = merge_opt(shared.min_uniform_edges, cfg.min_uniform_edges, "min-uniform-edges")
        .unwrap_or(2);
    if min_uniform_edges == 0 {
        // 0 would let top-heavy pairing start before any results exist —
        // selection weights are only derived after a completed refit.
        bail("min-uniform-edges must be at least 1");
    }
    let judgements_per_refit = merge_opt(shared.judgements_per_refit, cfg.judgements_per_refit, "judgements-per-refit");
    if judgements_per_refit == Some(0) {
        bail("--judgements-per-refit must be at least 1");
    }
    let prior_tau2 = merge_opt(shared.prior_tau2, cfg.prior_tau2, "prior-tau2")
        .unwrap_or(DEFAULT_PRIOR_TAU2);
    if !prior_tau2.is_finite() || prior_tau2 <= 0.0 {
        bail(format!("prior-tau2={prior_tau2}, must be finite and > 0"));
    }
    let bias_prior_tau2 = merge_opt(shared.bias_prior_tau2, cfg.bias_prior_tau2, "bias-prior-tau2")
        .unwrap_or(DEFAULT_BIAS_PRIOR_TAU2);
    if !bias_prior_tau2.is_finite() || bias_prior_tau2 <= 0.0 {
        bail(format!("bias-prior-tau2={bias_prior_tau2}, must be finite and > 0"));
    }
    let live_top = merge_opt(shared.live_top, cfg.live_top, "live-top");
    let emit_interim_rankings = shared.emit_interim_rankings.unwrap_or(false);
    let save_successful_judgements = match (shared.save_successful_judgements.clone(), cfg.save_successful_judgements.clone()) {
        (Some(c), Some(f)) => {
            if c != f {
                eprintln!("Warning: --save-successful-judgements ({}) overrides config file value ({})",
                    c.display(), f.display());
            }
            Some(c)
        }
        (c @ Some(_), None) => c,
        (None, f) => f,
    };
    let include_successful_prompts = merge_opt(shared.include_successful_prompts, cfg.include_successful_prompts, "include-successful-prompts")
        .unwrap_or(false);
    if include_successful_prompts && save_successful_judgements.is_none() {
        bail("--include-successful-prompts requires --save-successful-judgements");
    }
    let output_format = merge_opt(shared.output_format, cfg.output_format, "output-format").unwrap_or_else(|| {
        if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            OutputFormat::Table
        } else {
            OutputFormat::Json
        }
    });
    let verbose = merge_opt(shared.verbose, cfg.verbose, "verbose").unwrap_or(false);
    let save_failed_judgements = match (shared.save_failed_judgements.clone(), cfg.save_failed_judgements.clone()) {
        (Some(c), Some(f)) => {
            if c != f {
                eprintln!("Warning: --save-failed-judgements ({}) overrides config file value ({})",
                    c.display(), f.display());
            }
            Some(c)
        }
        (c @ Some(_), None) => c,
        (None, f) => f,
    };

    let seed = merge_opt(shared.seed, cfg.seed, "seed");

    // bias_prior: user specifies in probability space, we convert to logit
    let bias_prior = merge_opt(shared.bias_prior, cfg.bias_prior, "bias-prior")
        .unwrap_or(DEFAULT_BIAS_PRIOR);
    if !bias_prior.is_finite() || bias_prior <= 0.0 || bias_prior >= 1.0 {
        bail("--bias-prior must be greater than 0.0 and less than 1.0");
    }
    let bias_prior_logit = (bias_prior / (1.0 - bias_prior)).ln();

    let reasoning_enabled = merge_opt(shared.reasoning, cfg.reasoning_enabled, "reasoning")
        .unwrap_or(true);

    if !reasoning_enabled && shared.analysis_length.is_some() {
        eprintln!("Warning: --analysis-length is ignored when reasoning is disabled");
    }

    // Prompt template: CLI path > config path > built-in default
    let prompt_template = {
        let cli_path = shared.prompt_template.clone();
        let cfg_path = cfg.prompt_template.as_ref().map(PathBuf::from);

        if let (Some(cp), Some(fp)) = (&cli_path, &cfg_path)
            && cp != fp
        {
            eprintln!("Warning: --prompt-template ({}) overrides config file value ({})",
                cp.display(), fp.display());
        }

        let template_path = cli_path.or(cfg_path);
        match template_path {
            Some(path) if uses_lineups => prompt::load_lineup_template(&path, lineup_size),
            Some(path) => prompt::load_template(&path, reasoning_enabled),
            None if uses_lineups && reasoning_enabled => prompt::default_lineup_template(lineup_size),
            None if uses_lineups => prompt::default_lineup_template_no_reasoning(lineup_size),
            None if reasoning_enabled => prompt::DEFAULT_TEMPLATE.to_string(),
            None => prompt::DEFAULT_TEMPLATE_NO_REASONING.to_string(),
        }
    };

    ResolvedConfig {
        judgements_per_item,
        judgement_distribution,
        lineup_size,
        selection_sharpness,
        anchor_index,
        selection_cutoff,
        selection_coverage,
        target_prior_edges,
        stop_confidence,
        retries,
        analysis_length,
        reasoning_enabled,
        prompt_template,
        confidence_level,
        regularization_strength,
        bias_prior_logit,
        matchmaking_sharpness,
        min_uniform_edges,
        judgements_per_refit,
        prior_tau2,
        bias_prior_tau2,
        live_top,
        emit_interim_rankings,
        save_successful_judgements,
        include_successful_prompts,
        output_format,
        verbose,
        save_failed_judgements,
        seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{JudgeConfig, NanojudgeConfig};

    fn default_cli() -> ConfigArgs {
        ConfigArgs {
            api_key: None,
            logprobs: None,
            judgements_per_item: None,
            concurrency: None,
            min_logprob_coverage: None,
            verdict_temperature: None,
            judgement_distribution: None,
            lineup_size: None,
            selection_sharpness: None,
            anchor_index: None,
            cutoff: None,
            coverage: None,
            target_prior_edges: None,
            stop_confidence: None,
            retries: None,
            analysis_length: None,
            reasoning: None,
            prompt_template: None,
            confidence_level: None,
            regularization_strength: None,
            bias_prior: None,
            matchmaking_sharpness: None,
            min_uniform_edges: None,
            judgements_per_refit: None,
            prior_tau2: None,
            bias_prior_tau2: None,
            live_top: None,
            emit_interim_rankings: None,
            save_successful_judgements: None,
            include_successful_prompts: None,
            output_format: None,
            verbose: None,
            save_failed_judgements: None,
            seed: None,
        }
    }

    fn one_judge_config() -> NanojudgeConfig {
        NanojudgeConfig {
            judge: Some(vec![JudgeConfig {
                endpoint: "http://localhost:8000".into(),
                model: "test-model".into(),
                concurrency: None,
                weight: None,
                temperature: Some(0.7),
                temperature_jitter: None,
                presence_penalty: None,
                top_p: None,
                min_logprob_coverage: None,
                verdict_temperature: None,
                api_key_env: None,
                max_tokens: None,
                reasoning_effort: None,
                chat_template_kwargs: None,
            }]),
            ..Default::default()
        }
    }

    #[test]
    fn test_min_logprob_coverage_default() {
        let cli = default_cli();
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].min_logprob_coverage, parse::DEFAULT_MIN_LOGPROB_COVERAGE);
    }

    #[test]
    fn test_min_logprob_coverage_from_cli() {
        let mut cli = default_cli();
        cli.min_logprob_coverage = Some(0.9);
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].min_logprob_coverage, 0.9);
    }

    #[test]
    fn test_min_logprob_coverage_from_config() {
        let cli = default_cli();
        let mut cfg = one_judge_config();
        cfg.min_logprob_coverage = Some(0.85);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].min_logprob_coverage, 0.85);
    }

    #[test]
    fn test_min_logprob_coverage_cli_overrides_config() {
        let mut cli = default_cli();
        cli.min_logprob_coverage = Some(0.9);
        let mut cfg = one_judge_config();
        cfg.min_logprob_coverage = Some(0.85);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].min_logprob_coverage, 0.9);
    }

    #[test]
    fn test_min_logprob_coverage_per_judge_overrides_global() {
        let mut cli = default_cli();
        cli.min_logprob_coverage = Some(0.9);
        let mut cfg = one_judge_config();
        cfg.judge.as_mut().unwrap()[0].min_logprob_coverage = Some(0.7);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].min_logprob_coverage, 0.7);
    }

    #[test]
    fn test_verdict_temperature_default_reasoning() {
        let cli = default_cli();
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].verdict_temperature, DEFAULT_VERDICT_TEMPERATURE_REASONING);
    }

    #[test]
    fn test_verdict_temperature_default_no_reasoning() {
        let cli = default_cli();
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), false);
        assert_eq!(judges[0].verdict_temperature, DEFAULT_VERDICT_TEMPERATURE_NO_REASONING);
    }

    #[test]
    fn test_verdict_temperature_from_cli() {
        let mut cli = default_cli();
        cli.verdict_temperature = Some(6.0);
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].verdict_temperature, 6.0);
    }

    #[test]
    fn test_verdict_temperature_from_config() {
        let cli = default_cli();
        let mut cfg = one_judge_config();
        cfg.verdict_temperature = Some(2.0);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].verdict_temperature, 2.0);
    }

    #[test]
    fn test_verdict_temperature_cli_overrides_config() {
        let mut cli = default_cli();
        cli.verdict_temperature = Some(6.0);
        let mut cfg = one_judge_config();
        cfg.verdict_temperature = Some(2.0);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].verdict_temperature, 6.0);
    }

    #[test]
    fn test_verdict_temperature_per_judge_overrides_global() {
        let mut cli = default_cli();
        cli.verdict_temperature = Some(6.0);
        let mut cfg = one_judge_config();
        cfg.judge.as_mut().unwrap()[0].verdict_temperature = Some(2.5);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].verdict_temperature, 2.5);
    }

    #[test]
    fn test_verdict_temperature_explicit_wins_in_no_reasoning_mode() {
        // An explicit global value applies as-is even when reasoning is off —
        // the mode-dependent default only kicks in when nothing is set.
        let cli = default_cli();
        let mut cfg = one_judge_config();
        cfg.verdict_temperature = Some(4.0);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), false);
        assert_eq!(judges[0].verdict_temperature, 4.0);
    }

    #[test]
    fn test_concurrency_from_cli() {
        let mut cli = default_cli();
        cli.concurrency = Some(32);
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].concurrency, 32);
    }

    #[test]
    fn test_concurrency_cli_overrides_config() {
        let mut cli = default_cli();
        cli.concurrency = Some(32);
        let mut cfg = one_judge_config();
        cfg.concurrency = Some(8);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert_eq!(judges[0].concurrency, 32);
    }

    #[test]
    fn test_logprobs_from_cli() {
        let mut cli = default_cli();
        cli.logprobs = Some(true);
        let cfg = one_judge_config();
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert!(judges[0].logprobs);
    }

    #[test]
    fn test_logprobs_cli_false_overrides_config() {
        let mut cli = default_cli();
        cli.logprobs = Some(false);
        let mut cfg = one_judge_config();
        cfg.logprobs = Some(true);
        let judges = resolve_judges(&cli, &cfg, Path::new("test.toml"), true);
        assert!(!judges[0].logprobs);
    }

    fn cli_with_budget() -> ConfigArgs {
        let mut cli = default_cli();
        cli.judgements_per_item = Some(10);
        cli
    }

    #[test]
    fn test_reasoning_from_cli() {
        let mut cli = cli_with_budget();
        cli.reasoning = Some(false);
        let cfg = NanojudgeConfig::default();
        let resolved = resolve_config(&cli, &cfg);
        assert!(!resolved.reasoning_enabled);
    }

    #[test]
    fn test_reasoning_cli_overrides_config() {
        let mut cli = cli_with_budget();
        cli.reasoning = Some(true);
        let cfg = NanojudgeConfig { reasoning_enabled: Some(false), ..Default::default() };
        let resolved = resolve_config(&cli, &cfg);
        assert!(resolved.reasoning_enabled);
    }

    #[test]
    fn test_reasoning_from_config() {
        let cli = cli_with_budget();
        let cfg = NanojudgeConfig { reasoning_enabled: Some(false), judgements_per_item: Some(10), ..Default::default() };
        let resolved = resolve_config(&cli, &cfg);
        assert!(!resolved.reasoning_enabled);
    }

    #[test]
    fn test_stop_confidence_absent_means_no_early_stop() {
        let cli = cli_with_budget();
        let cfg = NanojudgeConfig::default();
        let resolved = resolve_config(&cli, &cfg);
        assert_eq!(resolved.stop_confidence, None);
    }

    #[test]
    fn test_judgements_per_refit_is_literal() {
        let mut cli = cli_with_budget();
        cli.judgements_per_refit = Some(3);
        let resolved = resolve_config(&cli, &NanojudgeConfig::default());
        assert_eq!(resolved.judgements_per_refit, Some(3));
    }

    #[test]
    fn test_stop_confidence_from_cli_with_top_heavy() {
        let mut cli = cli_with_budget();
        cli.judgement_distribution = Some("top-heavy".into());
        cli.stop_confidence = Some(0.95);
        let cfg = NanojudgeConfig::default();
        let resolved = resolve_config(&cli, &cfg);
        assert_eq!(resolved.stop_confidence, Some(0.95));
    }

    #[test]
    fn test_stop_confidence_cli_overrides_config() {
        let mut cli = cli_with_budget();
        cli.judgement_distribution = Some("top-heavy".into());
        cli.stop_confidence = Some(0.99);
        let cfg = NanojudgeConfig { stop_confidence: Some(0.9), ..Default::default() };
        let resolved = resolve_config(&cli, &cfg);
        assert_eq!(resolved.stop_confidence, Some(0.99));
    }

    #[test]
    fn test_judgement_distribution_defaults_to_top_heavy() {
        let cli = cli_with_budget();
        let cfg = NanojudgeConfig::default();
        let resolved = resolve_config(&cli, &cfg);
        assert_eq!(resolved.judgement_distribution, JudgementDistribution::TopHeavy);
    }
}
