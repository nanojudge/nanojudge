/// nanojudge-bench: Synthetic benchmark harness for NanoJudge ranking accuracy.
///
/// Stands up a fake OpenAI-compatible endpoint backed by a secret strength table,
/// points the real NanoJudge CLI at it, and measures how well the engine recovers
/// the true item ordering across many independent trials.
mod metrics;
mod server;
mod trial;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "nanojudge-bench",
    about = "Synthetic benchmark for NanoJudge ranking accuracy"
)]
struct Args {
    /// Number of synthetic items per trial.
    #[arg(short = 'n', long)]
    items: usize,

    /// Comparison rounds per trial.
    #[arg(short, long)]
    rounds: usize,

    /// Number of independent trials to run.
    #[arg(short, long)]
    trials: usize,

    /// Standard deviation of the true strength distribution (normal, mean 0).
    #[arg(long)]
    spread: f64,

    /// How many top positions to include in the displacement metric.
    #[arg(long)]
    report_top_k: usize,

    /// Master RNG seed (omit for OS entropy).
    #[arg(long)]
    seed: Option<u64>,

    /// Comparison distribution: "uniform" or "top-heavy".
    #[arg(long)]
    comparison_distribution: String,

    /// Use logprobs mode instead of text-verdict mode.
    #[arg(long)]
    logprobs: bool,

    /// Path to the nanojudge binary (auto-detected from sibling binary if omitted).
    #[arg(long)]
    bin: Option<PathBuf>,

    /// Print per-trial results to stderr.
    #[arg(short, long)]
    verbose: bool,
}

fn find_nanojudge_binary(override_path: Option<PathBuf>) -> PathBuf {
    if let Some(path) = override_path {
        if !path.exists() {
            eprintln!("Error: specified binary not found: {}", path.display());
            std::process::exit(1);
        }
        return path;
    }

    let bench_exe = std::env::current_exe().expect("failed to locate current executable");
    let bin_dir = bench_exe.parent().expect("failed to get binary directory");
    let candidate = bin_dir.join("nanojudge");

    if !candidate.exists() {
        eprintln!(
            "Error: nanojudge binary not found at {}\n\
             Build it first with: cargo build --bin nanojudge",
            candidate.display()
        );
        std::process::exit(1);
    }

    candidate
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if args.items < 2 {
        eprintln!("Error: need at least 2 items");
        std::process::exit(1);
    }
    if args.rounds == 0 {
        eprintln!("Error: need at least 1 round");
        std::process::exit(1);
    }
    if args.trials == 0 {
        eprintln!("Error: need at least 1 trial");
        std::process::exit(1);
    }
    if !args.spread.is_finite() || args.spread <= 0.0 {
        eprintln!("Error: --spread must be a positive finite number");
        std::process::exit(1);
    }
    if args.comparison_distribution != "uniform" && args.comparison_distribution != "top-heavy" {
        eprintln!("Error: --comparison-distribution must be \"uniform\" or \"top-heavy\"");
        std::process::exit(1);
    }

    let nanojudge_bin = find_nanojudge_binary(args.bin);

    let master_seed = args.seed.unwrap_or_else(rand::random::<u64>);
    let top_k = args.report_top_k.min(args.items - 1).max(1);

    let comparisons_per_trial = (args.items / 2) * args.rounds;

    eprintln!("NanoJudge Synthetic Benchmark");
    eprintln!("  Items: {}", args.items);
    eprintln!("  Rounds: {}", args.rounds);
    eprintln!("  Trials: {}", args.trials);
    eprintln!("  Comparisons per trial: {}", comparisons_per_trial);
    eprintln!("  Spread: {:.1}", args.spread);
    eprintln!("  Logprobs: {}", args.logprobs);
    eprintln!("  Report top-K: {}", top_k);
    eprintln!("  Distribution: {}", args.comparison_distribution);
    eprintln!("  Seed: {}", master_seed);
    eprintln!();

    let config = trial::TrialConfig {
        num_items: args.items,
        rounds: args.rounds,
        strength_spread: args.spread,
        use_logprobs: args.logprobs,
        distribution: &args.comparison_distribution,
        nanojudge_bin: &nanojudge_bin,
    };

    let mut results: Vec<trial::TrialResult> = Vec::with_capacity(args.trials);
    let mut errors = 0usize;

    for t in 0..args.trials {
        let trial_seed = master_seed.wrapping_add(t as u64 * 1000);

        if !args.verbose {
            eprint!("\r  Running trial {}/{}...", t + 1, args.trials);
        }

        match trial::run(&config, trial_seed, top_k).await {
            Ok(result) => {
                if args.verbose {
                    eprintln!(
                        "  Trial {:>4}/{}: rho={:.4}  top1-off={:.1}  top{}-off={:.2}  comparisons={}  {:.1}s",
                        t + 1,
                        args.trials,
                        result.spearman_rho,
                        result.top_1_displacement,
                        top_k,
                        result.top_k_displacement,
                        result.comparisons,
                        result.duration.as_secs_f64(),
                    );
                }
                results.push(result);
            }
            Err(e) => {
                errors += 1;
                eprintln!(
                    "\r  Trial {:>4}/{} FAILED: {}",
                    t + 1,
                    args.trials,
                    e
                );
            }
        }
    }

    if !args.verbose {
        eprint!("\r{}\r", " ".repeat(60));
    }

    if results.is_empty() {
        eprintln!("All {} trials failed.", args.trials);
        std::process::exit(1);
    }

    print_summary(&results, top_k, errors);
}

// ---------------------------------------------------------------------------
// Summary output
// ---------------------------------------------------------------------------

fn print_summary(results: &[trial::TrialResult], top_k: usize, errors: usize) {
    let rhos: Vec<f64> = results.iter().map(|r| r.spearman_rho).collect();
    let top1_disp: Vec<f64> = results.iter().map(|r| r.top_1_displacement).collect();
    let topk_disp: Vec<f64> = results.iter().map(|r| r.top_k_displacement).collect();
    let times: Vec<f64> = results.iter().map(|r| r.duration.as_secs_f64()).collect();

    println!("Results ({} trials, {} errors)", results.len(), errors);
    println!();
    println!(
        "  {:<20} {:>8} {:>8} {:>8} {:>8}",
        "Metric", "Mean", "Std", "Min", "Max"
    );
    println!(
        "  {:<20} {:>8} {:>8} {:>8} {:>8}",
        "------", "----", "---", "---", "---"
    );
    println!(
        "  {:<20} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
        "Spearman rho",
        mean(&rhos),
        std_dev(&rhos),
        fmin(&rhos),
        fmax(&rhos),
    );
    println!(
        "  {:<20} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
        "Top-1 displacement",
        mean(&top1_disp),
        std_dev(&top1_disp),
        fmin(&top1_disp),
        fmax(&top1_disp),
    );
    println!(
        "  {:<20} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
        format!("Top-{top_k} displacement"),
        mean(&topk_disp),
        std_dev(&topk_disp),
        fmin(&topk_disp),
        fmax(&topk_disp),
    );
    println!(
        "  {:<20} {:>8.2}s {:>7.2}s {:>7.2}s {:>7.2}s",
        "Time/trial",
        mean(&times),
        std_dev(&times),
        fmin(&times),
        fmax(&times),
    );
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn fmin(v: &[f64]) -> f64 {
    v.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

fn fmax(v: &[f64]) -> f64 {
    v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}
