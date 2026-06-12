use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use nanojudge_core::{ComparisonDistribution, stable_hash};

use crate::args::{ConfigArgs, OutputFormat};
use crate::bail;
use crate::config;
use crate::parse;
use crate::prompt;

const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_TEMPERATURE_JITTER: f64 = 0.0;
const DEFAULT_MAX_RETRIES: usize = 3;
const DEFAULT_ANALYSIS_LENGTH: &str = "2 paragraphs";

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
    pub rounds: Option<usize>,
    pub comparisons: Option<usize>,
    pub comparison_distribution: ComparisonDistribution,
    pub top_k: Option<usize>,
    pub retries: usize,
    pub analysis_length: String,
    pub reasoning_enabled: bool,
    pub prompt_template: String,
    pub confidence_level: f64,
    pub regularization_strength: f64,
    pub mcmc_iterations: usize,
    pub mcmc_burn_in: usize,
    pub bias_prior_logit: f64,
    pub matchmaking_sharpness: f64,
    pub min_uniform_games: usize,
    pub prior_tau2: f64,
    pub proposal_std: f64,
    pub bias_prior_tau2: f64,
    pub bias_proposal_std: f64,
    pub gap_proposal_std: f64,
    pub live_top: Option<usize>,
    pub save_comparisons: Option<PathBuf>,
    pub output_format: OutputFormat,
    pub verbose: bool,
    pub save_failures: Option<PathBuf>,
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
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    pub judge_id: u64,
    pub display_name: String,
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

    let global_min_logprob_coverage = merge_opt(shared.min_logprob_coverage, cfg.min_logprob_coverage, "min-logprob-coverage")
        .unwrap_or(parse::DEFAULT_MIN_LOGPROB_COVERAGE);
    if !global_min_logprob_coverage.is_finite() || global_min_logprob_coverage <= 0.0 || global_min_logprob_coverage > 1.0 {
        bail(format!(
            "min_logprob_coverage={global_min_logprob_coverage}, must be finite, > 0.0 and <= 1.0"
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

    // Compute display names: model is default, disambiguate with endpoint host if models collide
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
            let host = jc.endpoint
                .trim_start_matches("http://")
                .trim_start_matches("https://")
                .split('/')
                .next()
                .unwrap_or(&jc.endpoint);
            format!("{} ({})", jc.model, host)
        } else {
            jc.model.clone()
        };

        let judge_id = stable_hash(&format!("{}\0{}", jc.endpoint, jc.model));
        let weight = jc.weight.unwrap_or(1.0);
        if !weight.is_finite() || weight <= 0.0 {
            bail(format!("Judge {} has non-positive weight {}. All weights must be > 0.", jc.model, weight));
        }

        let min_logprob_coverage = jc.min_logprob_coverage.unwrap_or(global_min_logprob_coverage);
        if !min_logprob_coverage.is_finite() || min_logprob_coverage <= 0.0 || min_logprob_coverage > 1.0 {
            bail(format!(
                "Judge {} has min_logprob_coverage={}, must be finite, > 0.0 and <= 1.0",
                jc.model, min_logprob_coverage
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
            concurrency: jc.concurrency.unwrap_or(global_concurrency),
            weight,
            min_logprob_coverage,
            max_tokens: jc.max_tokens.unwrap_or(default_max_tokens),
            reasoning_effort: jc.reasoning_effort.clone(),
            judge_id,
            display_name,
        });
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
    let rounds = merge_opt(shared.rounds, cfg.rounds, "rounds");
    let comparisons = merge_opt(shared.comparisons, cfg.comparisons, "comparisons");

    if rounds.is_some() && comparisons.is_some() {
        bail("Specify --rounds or --comparisons, not both.");
    }

    let comparison_distribution_str = merge_opt(shared.comparison_distribution.clone(), cfg.comparison_distribution.clone(), "comparison-distribution")
        .unwrap_or_else(|| "uniform".to_string());
    let comparison_distribution = match comparison_distribution_str.as_str() {
        "uniform" => ComparisonDistribution::Uniform,
        "top-heavy" => ComparisonDistribution::TopHeavy,
        other => bail(format!("Unknown comparison distribution \"{other}\". Use \"uniform\" or \"top-heavy\".")),
    };

    let top_k = merge_opt(shared.top_k, cfg.top_k, "top-k");
    let retries = merge_opt(shared.retries, cfg.retries, "retries")
        .unwrap_or(DEFAULT_MAX_RETRIES);
    let analysis_length = merge_opt(shared.analysis_length.clone(), cfg.analysis_length.clone(), "analysis-length")
        .unwrap_or_else(|| DEFAULT_ANALYSIS_LENGTH.to_string());

    let confidence_level = merge_opt(shared.confidence_level, cfg.confidence_level, "confidence-level")
        .unwrap_or(0.95);
    let regularization_strength = merge_opt(shared.regularization_strength, cfg.regularization_strength, "regularization-strength")
        .unwrap_or(0.01);
    let mcmc_iterations = merge_opt(shared.mcmc_iterations, cfg.mcmc_iterations, "mcmc-iterations")
        .unwrap_or(2000);
    let mcmc_burn_in = merge_opt(shared.mcmc_burn_in, cfg.mcmc_burn_in, "mcmc-burn-in")
        .unwrap_or(500);
    let matchmaking_sharpness = merge_opt(shared.matchmaking_sharpness, cfg.matchmaking_sharpness, "matchmaking-sharpness")
        .unwrap_or(1.0);
    let min_uniform_games = merge_opt(shared.min_uniform_games, cfg.min_uniform_games, "min-uniform-games")
        .unwrap_or(3);
    let prior_tau2 = merge_opt(shared.prior_tau2, cfg.prior_tau2, "prior-tau2")
        .unwrap_or(10.0);
    let proposal_std = merge_opt(shared.proposal_std, cfg.proposal_std, "proposal-std")
        .unwrap_or(0.3);
    let bias_prior_tau2 = merge_opt(shared.bias_prior_tau2, cfg.bias_prior_tau2, "bias-prior-tau2")
        .unwrap_or(2.0);
    let bias_proposal_std = merge_opt(shared.bias_proposal_std, cfg.bias_proposal_std, "bias-proposal-std")
        .unwrap_or(0.15);
    let gap_proposal_std = merge_opt(shared.gap_proposal_std, cfg.gap_proposal_std, "gap-proposal-std")
        .unwrap_or(0.15);

    let live_top = merge_opt(shared.live_top, cfg.live_top, "live-top");
    let save_comparisons = match (shared.save_comparisons.clone(), cfg.save_comparisons.clone()) {
        (Some(c), Some(f)) => {
            if c != f {
                eprintln!("Warning: --save-comparisons ({}) overrides config file value ({})",
                    c.display(), f.display());
            }
            Some(c)
        }
        (c @ Some(_), None) => c,
        (None, f) => f,
    };
    let output_format = merge_opt(shared.output_format, cfg.output_format, "output-format").unwrap_or(OutputFormat::Table);
    let verbose = merge_opt(shared.verbose, cfg.verbose, "verbose").unwrap_or(false);
    let save_failures = match (shared.save_failures.clone(), cfg.save_failures.clone()) {
        (Some(c), Some(f)) => {
            if c != f {
                eprintln!("Warning: --save-failures ({}) overrides config file value ({})",
                    c.display(), f.display());
            }
            Some(c)
        }
        (c @ Some(_), None) => c,
        (None, f) => f,
    };

    // bias_prior: user specifies in probability space, we convert to logit
    let bias_prior = merge_opt(shared.bias_prior, cfg.bias_prior, "bias-prior")
        .unwrap_or(0.5);
    if bias_prior <= 0.0 || bias_prior >= 1.0 {
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

        if let (Some(cp), Some(fp)) = (&cli_path, &cfg_path) {
            if cp != fp {
                eprintln!("Warning: --prompt-template ({}) overrides config file value ({})",
                    cp.display(), fp.display());
            }
        }

        let template_path = cli_path.or(cfg_path);
        match template_path {
            Some(path) => prompt::load_template(&path, reasoning_enabled),
            None if reasoning_enabled => prompt::DEFAULT_TEMPLATE.to_string(),
            None => prompt::DEFAULT_TEMPLATE_NO_REASONING.to_string(),
        }
    };

    ResolvedConfig {
        rounds,
        comparisons,
        comparison_distribution,
        top_k,
        retries,
        analysis_length,
        reasoning_enabled,
        prompt_template,
        confidence_level,
        regularization_strength,
        mcmc_iterations,
        mcmc_burn_in,
        bias_prior_logit,
        matchmaking_sharpness,
        min_uniform_games,
        prior_tau2,
        proposal_std,
        bias_prior_tau2,
        bias_proposal_std,
        gap_proposal_std,
        live_top,
        save_comparisons,
        output_format,
        verbose,
        save_failures,
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
            rounds: None,
            comparisons: None,
            concurrency: None,
            min_logprob_coverage: None,
            comparison_distribution: None,
            top_k: None,
            retries: None,
            analysis_length: None,
            reasoning: None,
            prompt_template: None,
            confidence_level: None,
            regularization_strength: None,
            mcmc_iterations: None,
            mcmc_burn_in: None,
            bias_prior: None,
            matchmaking_sharpness: None,
            min_uniform_games: None,
            prior_tau2: None,
            proposal_std: None,
            bias_prior_tau2: None,
            bias_proposal_std: None,
            gap_proposal_std: None,
            live_top: None,
            save_comparisons: None,
            output_format: None,
            verbose: None,
            save_failures: None,
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
                api_key_env: None,
                max_tokens: None,
                reasoning_effort: None,
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

    #[test]
    fn test_reasoning_from_cli() {
        let mut cli = default_cli();
        cli.reasoning = Some(false);
        let cfg = NanojudgeConfig::default();
        let resolved = resolve_config(&cli, &cfg);
        assert!(!resolved.reasoning_enabled);
    }

    #[test]
    fn test_reasoning_cli_overrides_config() {
        let mut cli = default_cli();
        cli.reasoning = Some(true);
        let mut cfg = NanojudgeConfig::default();
        cfg.reasoning_enabled = Some(false);
        let resolved = resolve_config(&cli, &cfg);
        assert!(resolved.reasoning_enabled);
    }

    #[test]
    fn test_reasoning_from_config() {
        let cli = default_cli();
        let mut cfg = NanojudgeConfig::default();
        cfg.reasoning_enabled = Some(false);
        let resolved = resolve_config(&cli, &cfg);
        assert!(!resolved.reasoning_enabled);
    }
}
