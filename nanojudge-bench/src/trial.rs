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
    pub judgements_per_item: usize,
    pub actual_tau2: f64,
    pub prior_tau2: Option<f64>,
    pub samples_per_judgement: usize,
    pub judgement_distribution: &'a str,
    pub selection_sharpness: Option<f64>,
    pub anchor_index: Option<f64>,
    pub cutoff: Option<f64>,
    pub coverage: Option<f64>,
    pub target_prior_edges: Option<f64>,
    pub stop_confidence: Option<f64>,
    pub min_uniform_edges: Option<usize>,
    pub judgements_per_refit: Option<usize>,
    pub lineup_size: usize,
    pub nanojudge_bin: &'a Path,
}

/// Accuracy metrics for one refit's interim ranking, measured against ground
/// truth. One per refit when the CLI is run with `--emit-interim-rankings`.
pub struct InterimMetrics {
    pub judgements: usize,
    pub spearman_rho: f64,
    pub top_1_displacement: f64,
    pub top_k_displacement: f64,
}

pub struct TrialResult {
    pub spearman_rho: f64,
    pub top_1_displacement: f64,
    pub top_k_displacement: f64,
    pub judgements: usize,
    pub duration: std::time::Duration,
    /// Per-refit interim metrics, for plotting convergence.
    pub per_refit: Vec<InterimMetrics>,
    /// Rendered ranking table for this trial, present only when `capture_example`
    /// was set (we print one example and suppress the rest to avoid spam).
    pub example_ranking: Option<String>,
}

pub async fn run(
    config: &TrialConfig<'_>,
    seed: u64,
    top_k: usize,
    capture_example: bool,
) -> Result<TrialResult, String> {
    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);

    // Draw strengths from N(0, actual_tau2), then assign names by rank so that
    // item_0001 is the strongest, item_0002 the second strongest, and so on.
    // This makes ranking errors obvious by eye in the example table (a perfect
    // recovery is item_0001, item_0002, … in order).
    let strength_std = config.actual_tau2.sqrt();
    let mut sampled_strengths: Vec<f64> = (0..config.num_items)
        .map(|_| sample_normal(&mut rng) * strength_std)
        .collect();
    sampled_strengths.sort_by(|a, b| b.partial_cmp(a).unwrap()); // strongest first

    let mut strengths: HashMap<String, f64> = HashMap::with_capacity(config.num_items);
    let mut true_order: Vec<String> = Vec::with_capacity(config.num_items);
    for (i, &s) in sampled_strengths.iter().enumerate() {
        let name = format!("item_{:04}", i + 1);
        strengths.insert(name.clone(), s);
        true_order.push(name);
    }

    // Temp files for items and config. Dropped (deleted) at end of scope.
    let items_file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let config_file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;

    // Shuffle the order items are presented to the CLI so the input order
    // carries NO ground-truth signal. (true_order, used for metrics, stays in
    // strength order.) Without this, the CLI receives items in true-strength
    // order and any stable tie-break in its ranking defaults to the true order,
    // spuriously inflating accuracy when scores are tied (sparse early refits).
    let mut item_names: Vec<String> = (0..config.num_items)
        .map(|i| format!("item_{:04}", i + 1))
        .collect();
    use rand::seq::SliceRandom;
    item_names.shuffle(&mut rng);
    let items_text: String = item_names.join("\n");
    std::fs::write(items_file.path(), &items_text).map_err(|e| e.to_string())?;

    // Start fake server.
    let server_seed = seed.wrapping_add(1);
    let state = Arc::new(server::JudgeState {
        strengths,
        seed: server_seed,
        samples_per_judgement: config.samples_per_judgement,
        encounter_counts: std::sync::Mutex::new(std::collections::HashMap::new()),
    });
    let (port, server_handle) = server::start(state).await;

    // Guard ensures server is always cleaned up, even on early return.
    let _server_guard = ServerGuard(&server_handle);

    // The benchmark always uses the CLI's logprob-shaped transport so the fake
    // endpoint can pass empirical sample frequencies through as one soft
    // judgement. This is an implementation detail, not a benchmark mode:
    // `samples_per_judgement = 1` naturally produces a hard judgement.
    let config_toml = format!(
        "reasoning_enabled = false\n\
         logprobs = true\n\
         verdict_temperature = 1.0\n\
         \n\
         [[judge]]\n\
         endpoint = \"http://127.0.0.1:{port}\"\n\
         model = \"synthetic-judge\"\n\
         temperature = 0.0\n\
         concurrency = 1\n",
    );
    std::fs::write(config_file.path(), &config_toml).map_err(|e| e.to_string())?;

    // Derive a CLI seed from the trial seed so pairing is reproducible.
    let cli_seed = seed.wrapping_add(2);

    // Run the real CLI as a subprocess.
    let mut cmd = tokio::process::Command::new(config.nanojudge_bin);
    cmd.arg("rank")
        .arg("--items")
        .arg(items_file.path())
        .arg("--config")
        .arg(config_file.path())
        .arg("--criterion")
        .arg("Which is better?")
        .arg("--output-format")
        .arg("json")
        .arg("--judgements-per-item")
        .arg(config.judgements_per_item.to_string())
        .arg("--judgement-distribution")
        .arg(config.judgement_distribution)
        .arg("--seed")
        .arg(cli_seed.to_string())
        .arg("--emit-interim-rankings");

    // Forward top-heavy selection tuning if the bench was given it.
    if let Some(selection_sharpness) = config.selection_sharpness {
        cmd.arg("--selection-sharpness").arg(selection_sharpness.to_string());
    }
    if let Some(anchor_index) = config.anchor_index {
        cmd.arg("--anchor-index").arg(anchor_index.to_string());
    }
    if let Some(cutoff) = config.cutoff {
        cmd.arg("--cutoff").arg(cutoff.to_string());
    }
    if let Some(coverage) = config.coverage {
        cmd.arg("--coverage").arg(coverage.to_string());
    }
    if let Some(target_prior_edges) = config.target_prior_edges {
        cmd.arg("--target-prior-edges").arg(target_prior_edges.to_string());
    }
    if let Some(stop_confidence) = config.stop_confidence {
        cmd.arg("--stop-confidence").arg(stop_confidence.to_string());
    }
    if let Some(prior_tau2) = config.prior_tau2 {
        cmd.arg("--prior-tau2").arg(prior_tau2.to_string());
    }
    if let Some(min_uniform_edges) = config.min_uniform_edges {
        cmd.arg("--min-uniform-edges").arg(min_uniform_edges.to_string());
    }
    if let Some(judgements_per_refit) = config.judgements_per_refit {
        cmd.arg("--judgements-per-refit").arg(judgements_per_refit.to_string());
    }
    if config.lineup_size != 2 {
        cmd.arg("--lineup-size").arg(config.lineup_size.to_string());
    }

    let output = cmd
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

    let judgements = json["total_judgements"].as_u64().unwrap_or(0) as usize;

    // Render the example ranking table only for the trial that asked for it.
    let example_ranking = if capture_example {
        Some(render_ranking_table(items))
    } else {
        None
    };

    let duration = start.elapsed();

    // Compute metrics against ground truth.
    let spearman_rho = metrics::spearman_rho(&true_order, &output_order);
    let top_1_displacement = metrics::top_k_displacement(&true_order, &output_order, 1);
    let top_k_displacement = metrics::top_k_displacement(&true_order, &output_order, top_k);

    let interim_rankings = json["interim_rankings"]
        .as_array()
        .ok_or("missing 'interim_rankings' in output (was --emit-interim-rankings honored?)")?;
    let per_refit: Vec<InterimMetrics> = interim_rankings
        .iter()
        .map(|entry| {
            let judgements = entry["judgements"].as_u64().unwrap_or(0) as usize;
            let order: Vec<String> = entry["order"]
                .as_array()
                .ok_or("refit entry missing 'order' array")?
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            Ok(InterimMetrics {
                judgements,
                spearman_rho: metrics::spearman_rho(&true_order, &order),
                top_1_displacement: metrics::top_k_displacement(&true_order, &order, 1),
                top_k_displacement: metrics::top_k_displacement(&true_order, &order, top_k),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(TrialResult {
        spearman_rho,
        top_1_displacement,
        top_k_displacement,
        judgements,
        duration,
        per_refit,
        example_ranking,
    })
}

/// Render the CLI's JSON `items` array into a human-readable ranking table,
/// mirroring the CLI's own table columns (rank, item, score, 95% CI, per-item
/// edge count, id).
fn render_ranking_table(items: &[serde_json::Value]) -> String {
    let name_width = items
        .iter()
        .map(|it| it["name"].as_str().unwrap_or("").len())
        .max()
        .unwrap_or(4)
        .max(4);

    let mut out = String::new();
    out.push_str(&format!(
        " # | {:<name_width$} |   Score | 95% CI Low | 95% CI High | Edges | ID\n",
        "Item",
    ));
    out.push_str(&format!(
        "---|-{}-|---------|------------|-------------|-------|----\n",
        "-".repeat(name_width)
    ));
    for it in items {
        out.push_str(&format!(
            "{:>2} | {:<name_width$} | {:>7.4} | {:>10.2} | {:>11.2} | {:>5} | {:>2}\n",
            it["rank"].as_u64().unwrap_or(0),
            it["name"].as_str().unwrap_or(""),
            it["score"].as_f64().unwrap_or(0.0),
            it["lower_bound"].as_f64().unwrap_or(0.0),
            it["upper_bound"].as_f64().unwrap_or(0.0),
            it["edges"].as_u64().unwrap_or(0),
            it["id"].as_i64().unwrap_or(0),
        ));
    }
    out
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
