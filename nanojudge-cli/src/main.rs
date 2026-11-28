use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;

use nanojudge_cli::args::{Cli, Commands};
use nanojudge_cli::{Reporter, StderrReporter, benchmark, config, output, probe, rank, score};

fn bail(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let reporter: Arc<dyn Reporter> = Arc::new(StderrReporter);

    let result = match cli.command {
        Commands::Rank(args) => match rank::prepare(args, reporter).await {
            Ok(prepared) => {
                let format = prepared.output_format();
                let cancelled = Arc::new(AtomicBool::new(false));
                {
                    let cancelled = cancelled.clone();
                    tokio::spawn(async move {
                        let _ = tokio::signal::ctrl_c().await;
                        eprintln!("\nCancelling... (press Ctrl-C again to force quit)");
                        cancelled.store(true, Ordering::Relaxed);
                        let _ = tokio::signal::ctrl_c().await;
                        std::process::exit(130);
                    });
                }
                prepared.run(cancelled).await.map(|out| output::print(&out, format))
            }
            Err(e) => Err(e),
        },
        Commands::Score(args) => {
            let format = args.output_format.unwrap_or_else(output::default_format);
            score::run(args, reporter.as_ref()).map(|out| output::print(&out, format))
        }
        Commands::Benchmark(args) => benchmark::run_with(args, reporter, benchmark::print_report)
            .await
            .map(|_| ()),
        Commands::Probe(args) => probe::run(args, reporter.as_ref()).await.map(|_| ()),
        Commands::Init => config::create_default_config().map(|path| {
            println!("Created config at {}", path.display());
            println!("Edit it to set your default endpoint, model, etc.");
        }),
    };

    if let Err(e) = result {
        bail(e);
    }
}
