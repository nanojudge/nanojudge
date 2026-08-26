mod args;
mod benchmark;
mod config;
mod items;
mod llm;
mod output;
mod parse;
mod prompt;
mod rank;
mod resolve;
mod score;

use clap::Parser;

use crate::args::{BenchmarkArgs, Cli, Commands};
use crate::llm::LlmConfig;
use crate::resolve::{resolve_config, resolve_judges};

const DEFAULT_BENCHMARK_PAIRS: &str = "100";

const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
const DEFAULT_REGULARIZATION_STRENGTH: f64 = 0.01;
const DEFAULT_PRIOR_TAU2: f64 = 10.0;
const DEFAULT_BIAS_PRIOR_TAU2: f64 = 2.0;
const DEFAULT_BIAS_PRIOR: f64 = 0.5;

pub fn bail(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Rank(args) => rank::run(args).await,
        Commands::Score(args) => score::run(args),
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
    let resolved = resolve_config(&args.cfg, &cfg);
    let judges = resolve_judges(&args.cfg, &cfg, &config_path, resolved.reasoning_enabled);

    if args.num_pairs == 0 {
        bail("--num-pairs must be at least 1");
    }

    let template = &resolved.prompt_template;

    for judge in &judges {
        let llm_config = LlmConfig {
            endpoint: judge.endpoint.clone(),
            model: judge.model.clone(),
            api_key: judge.api_key.clone(),
            temperature: judge.temperature,
            temperature_jitter: judge.temperature_jitter,
            presence_penalty: judge.presence_penalty,
            top_p: judge.top_p,
            logprobs: judge.logprobs,
            max_tokens: judge.max_tokens,
            reasoning_effort: judge.reasoning_effort.clone(),
            chat_template_kwargs: judge.chat_template_kwargs.clone(),
        };

        benchmark::run_benchmark(
            &llm_config,
            &judge.display_name,
            args.num_pairs,
            judge.concurrency,
            judge.min_logprob_coverage,
            template,
            resolved.seed,
        ).await;

        if judges.len() > 1 {
            eprintln!();
        }
    }
}
