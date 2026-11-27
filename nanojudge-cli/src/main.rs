use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;

use nanojudge_cli::args::{Cli, Commands};
use nanojudge_cli::{benchmark, config, probe, rank, score};

fn bail(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Rank(args) => match rank::prepare(args).await {
            Ok(prepared) => {
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
                prepared.run(cancelled).await
            }
            Err(e) => Err(e),
        },
        Commands::Score(args) => score::run(args),
        Commands::Benchmark(args) => benchmark::run(args).await,
        Commands::Probe(args) => probe::run(args).await,
        Commands::Init => config::create_default_config().map(|path| {
            println!("Created config at {}", path.display());
            println!("Edit it to set your default endpoint, model, etc.");
        }),
    };

    if let Err(e) = result {
        bail(e);
    }
}
