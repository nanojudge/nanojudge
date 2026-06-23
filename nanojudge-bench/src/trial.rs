/// Single benchmark trial: generate items, start fake server, run the real CLI,
/// and compare the output ranking against ground truth.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::metrics;
use crate::server;

pub struct TrialConfig<'a> {
    pub num_items: usize,
    pub rounds: usize,
    pub strength_spread: f64,
    pub use_logprobs: bool,
    pub distribution: &'a str,
    pub nanojudge_bin: &'a Path,
}

pub struct TrialResult {
    pub spearman_rho: f64,
    pub top_1_displacement: f64,
    pub top_k_displacement: f64,
    pub comparisons: usize,
    pub duration: std::time::Duration,
}

pub async fn run(
    config: &TrialConfig<'_>,
    seed: u64,
    top_k: usize,
) -> Result<TrialResult, String> {
    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate items with true strengths drawn from N(0, spread).
    let mut strengths: HashMap<String, f64> = HashMap::with_capacity(config.num_items);
    for i in 0..config.num_items {
        let name = format!("item_{i:04}");
        let s: f64 = sample_normal(&mut rng) * config.strength_spread;
        strengths.insert(name, s);
    }

    // True ranking: items sorted by strength, strongest first.
    let mut true_order: Vec<(String, f64)> = strengths
        .iter()
        .map(|(name, &s)| (name.clone(), s))
        .collect();
    true_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let true_order: Vec<String> = true_order.into_iter().map(|(name, _)| name).collect();

    // Temp files for items and config. Dropped (deleted) at end of scope.
    let items_file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let config_file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;

    let items_text: String = (0..config.num_items)
        .map(|i| format!("item_{i:04}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(items_file.path(), &items_text).map_err(|e| e.to_string())?;

    // Start fake server.
    let server_seed = seed.wrapping_add(1);
    let state = Arc::new(server::JudgeState {
        strengths,
        seed: server_seed,
        pair_counts: std::sync::Mutex::new(std::collections::HashMap::new()),
    });
    let (port, server_handle) = server::start(state).await;

    // Guard ensures server is always cleaned up, even on early return.
    let _server_guard = ServerGuard(&server_handle);

    let config_toml = format!(
        "reasoning_enabled = false\n\
         logprobs = {logprobs}\n\
         \n\
         [[judge]]\n\
         endpoint = \"http://127.0.0.1:{port}\"\n\
         model = \"synthetic-judge\"\n\
         temperature = 0.0\n\
         concurrency = 1\n",
        logprobs = config.use_logprobs,
    );
    std::fs::write(config_file.path(), &config_toml).map_err(|e| e.to_string())?;

    // Derive a CLI seed from the trial seed so pairing + MCMC are reproducible.
    let cli_seed = seed.wrapping_add(2);

    // Run the real CLI as a subprocess.
    let output = tokio::process::Command::new(config.nanojudge_bin)
        .arg("rank")
        .arg("--items")
        .arg(items_file.path())
        .arg("--config")
        .arg(config_file.path())
        .arg("--criterion")
        .arg("Which is better?")
        .arg("--output-format")
        .arg("json")
        .arg("--rounds")
        .arg(config.rounds.to_string())
        .arg("--comparison-distribution")
        .arg(config.distribution)
        .arg("--seed")
        .arg(cli_seed.to_string())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("failed to run nanojudge: {e}"))?;

    drop(_server_guard);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nanojudge exited with {}: {stderr}", output.status));
    }

    // Parse the CLI's JSON output.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("bad JSON output: {e}"))?;

    let items = json["items"]
        .as_array()
        .ok_or("missing 'items' in output")?;

    let output_order: Vec<String> = items
        .iter()
        .map(|item| item["name"].as_str().unwrap().to_string())
        .collect();

    let comparisons = json["total_comparisons"].as_u64().unwrap_or(0) as usize;

    let duration = start.elapsed();

    // Compute metrics against ground truth.
    let spearman_rho = metrics::spearman_rho(&true_order, &output_order);
    let top_1_displacement = metrics::top_k_displacement(&true_order, &output_order, 1);
    let top_k_displacement = metrics::top_k_displacement(&true_order, &output_order, top_k);

    Ok(TrialResult {
        spearman_rho,
        top_1_displacement,
        top_k_displacement,
        comparisons,
        duration,
    })
}

struct ServerGuard<'a>(&'a tokio::task::JoinHandle<()>);

impl Drop for ServerGuard<'_> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Box-Muller transform for standard normal samples, avoiding a dependency on rand_distr.
fn sample_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-10);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
