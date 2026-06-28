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
use std::sync::Arc;

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
    strength_std: f64,

    /// How many top positions to include in the displacement metric.
    #[arg(long)]
    report_top_k: usize,

    /// Master RNG seed (omit for OS entropy).
    #[arg(long)]
    seed: Option<u64>,

    /// Comparison distribution: "uniform" or "top-heavy".
    #[arg(long)]
    comparison_distribution: String,

    /// Top-heavy selection sharpness forwarded to the CLI. Omit to use the CLI's
    /// default (0.1).
    #[arg(long)]
    selection_sharpness: Option<f64>,

    /// Top-heavy selection cutoff forwarded to the CLI. Omit to use the CLI's
    /// default (0 = no cutoff).
    #[arg(long)]
    cutoff: Option<f64>,

    /// Top-heavy coverage pull forwarded to the CLI. Omit to use the CLI's
    /// default (1 = proportional-fair).
    #[arg(long)]
    coverage: Option<f64>,

    /// Use logprobs mode instead of text-verdict mode.
    #[arg(long)]
    logprobs: bool,

    /// Path to the nanojudge binary (auto-detected from sibling binary if omitted).
    #[arg(long)]
    bin: Option<PathBuf>,

    /// How many trials to run concurrently. Each trial spawns its own CLI
    /// subprocess, so this parallelizes across cores. Omitted: half the logical
    /// cores (or 1 if that can't be determined).
    #[arg(long)]
    concurrency: Option<usize>,

    /// Print per-trial results to stderr.
    #[arg(short, long)]
    verbose: bool,
}

/// Current UTC time as "YYYY-MM-DD HH:MM:SS", using only std (no chrono
/// dependency) — mirrors the manual civil-date conversion used in
/// nanojudge-cli's benchmark log. UTC, so it reads ~1h behind a BST wall clock.
fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    let mut y = 1970i64;
    let mut remaining = days as i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining < md as i64 {
            mo = i + 1;
            break;
        }
        remaining -= md as i64;
    }
    let d = remaining + 1;
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, mo, d, h, m, s)
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

/// Owned trial configuration shared (via `Arc`) across the concurrent trial
/// tasks. Each task borrows a `trial::TrialConfig` from this for its run; owning
/// the `String`/`PathBuf` here is what lets the spawned tasks be `'static`.
struct SharedTrialConfig {
    num_items: usize,
    rounds: usize,
    strength_std: f64,
    use_logprobs: bool,
    distribution: String,
    selection_sharpness: Option<f64>,
    cutoff: Option<f64>,
    coverage: Option<f64>,
    nanojudge_bin: PathBuf,
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
    if !args.strength_std.is_finite() || args.strength_std <= 0.0 {
        eprintln!("Error: --strength-std must be a positive finite number");
        std::process::exit(1);
    }
    if args.comparison_distribution != "uniform" && args.comparison_distribution != "top-heavy" {
        eprintln!("Error: --comparison-distribution must be \"uniform\" or \"top-heavy\"");
        std::process::exit(1);
    }

    let nanojudge_bin = find_nanojudge_binary(args.bin);

    let master_seed = args.seed.unwrap_or_else(rand::random::<u64>);
    let top_k = args.report_top_k.min(args.items - 1).max(1);

    // Default concurrency: half the logical cores, or 1 if the count is
    // unavailable. At least 1 either way.
    let concurrency = args.concurrency.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(1)
    });
    if concurrency == 0 {
        eprintln!("Error: --concurrency must be at least 1");
        std::process::exit(1);
    }

    let comparisons_per_trial = (args.items / 2) * args.rounds;

    eprintln!("NanoJudge Synthetic Benchmark");
    eprintln!("  Items: {}", args.items);
    eprintln!("  Rounds: {}", args.rounds);
    eprintln!("  Trials: {}", args.trials);
    eprintln!("  Comparisons per trial: {}", comparisons_per_trial);
    eprintln!("  Strength std: {:.1}", args.strength_std);
    eprintln!("  Logprobs: {}", args.logprobs);
    eprintln!("  Report top-K: {}", top_k);
    eprintln!("  Distribution: {}", args.comparison_distribution);
    eprintln!("  Seed: {}", master_seed);
    eprintln!("  Concurrency: {}", concurrency);
    eprintln!();

    // Owned config the concurrent tasks share by Arc.
    let shared = Arc::new(SharedTrialConfig {
        num_items: args.items,
        rounds: args.rounds,
        strength_std: args.strength_std,
        use_logprobs: args.logprobs,
        distribution: args.comparison_distribution.clone(),
        selection_sharpness: args.selection_sharpness,
        cutoff: args.cutoff,
        coverage: args.coverage,
        nanojudge_bin: nanojudge_bin.clone(),
    });

    // Run trials concurrently, capped at `concurrency` by a semaphore. Trials are
    // independent (each owns its fake server on its own port + its own CLI
    // subprocess), so this just spreads them across cores. The first trial
    // (t == 0) captures the example ranking table.
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut join_set: tokio::task::JoinSet<(usize, Result<trial::TrialResult, String>)> =
        tokio::task::JoinSet::new();

    for t in 0..args.trials {
        let shared = shared.clone();
        let semaphore = semaphore.clone();
        let capture_example = t == 0;
        let trial_seed = master_seed.wrapping_add(t as u64 * 1000);
        join_set.spawn(async move {
            let _permit = semaphore
                .acquire()
                .await
                .expect("semaphore unexpectedly closed");
            let config = trial::TrialConfig {
                num_items: shared.num_items,
                rounds: shared.rounds,
                strength_std: shared.strength_std,
                use_logprobs: shared.use_logprobs,
                distribution: &shared.distribution,
                selection_sharpness: shared.selection_sharpness,
                cutoff: shared.cutoff,
                coverage: shared.coverage,
                nanojudge_bin: &shared.nanojudge_bin,
            };
            let res = trial::run(&config, trial_seed, top_k, capture_example).await;
            (t, res)
        });
    }

    let mut results: Vec<trial::TrialResult> = Vec::with_capacity(args.trials);
    let mut errors = 0usize;
    let mut completed = 0usize;

    // Trials complete out of order; collect as they finish (aggregate stats are
    // order-independent, and only t == 0 carries the example ranking).
    while let Some(joined) = join_set.join_next().await {
        let (t, res) = joined.expect("trial task panicked");
        completed += 1;
        match res {
            Ok(result) => {
                if args.verbose {
                    eprintln!(
                        "{}  Trial {:>4}/{}: rho={:.4}  top1-off={:.1}  top{}-off={:.2}  comparisons={}  {:.1}s",
                        utc_timestamp(),
                        t + 1,
                        args.trials,
                        result.spearman_rho,
                        result.top_1_displacement,
                        top_k,
                        result.top_k_displacement,
                        result.comparisons,
                        result.duration.as_secs_f64(),
                    );
                } else {
                    eprintln!(
                        "{}  Completed {}/{} (trial {})",
                        utc_timestamp(),
                        completed,
                        args.trials,
                        t + 1,
                    );
                }
                results.push(result);
            }
            Err(e) => {
                errors += 1;
                eprintln!("{}  Trial {} FAILED: {}", utc_timestamp(), t + 1, e);
            }
        }
    }

    if results.is_empty() {
        eprintln!("All {} trials failed.", args.trials);
        std::process::exit(1);
    }

    print_summary(&results, top_k, errors);

    // Show one example ranking (captured from the first trial) so the actual
    // output is visible, not just the aggregate metrics.
    if let Some(table) = results.iter().find_map(|r| r.example_ranking.as_ref()) {
        println!();
        println!("Example ranking (trial 1):");
        println!();
        print!("{table}");
    }
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
