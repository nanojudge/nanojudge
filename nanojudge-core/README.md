# nanojudge-core

Pure-computation ranking engine for weighted edges. Part of [nanojudge](https://github.com/nanojudge/nanojudge).

Takes edge win probabilities and produces ranked items with confidence intervals. No IO, no HTTP, no filesystem — just math. Bring your own source of judgements.

## Quick start

Items are identified by `i64` IDs — any unique numbers you want. The crate handles internal index mapping.

```rust
use nanojudge_core::{run_scoring, Edge, ScoringOptions, JudgeInfo, judge_hash};

let item_ids = vec![100, 200, 300];
let judge_id = judge_hash("my-endpoint", "my-model");

// category_probs = [P(item1 wins), P(item2 wins)]. new() records that
// item1 was shown first and item2 second, so scoring can correct for
// positional bias.
let edges = vec![
    Edge::new(100, 200, [0.8, 0.2], judge_id),
    Edge::new(200, 300, [0.7, 0.3], judge_id),
];

let judge_info = JudgeInfo { judge_ids: vec![judge_id], logprobs_mode: false };

let result = run_scoring(&item_ids, &edges, &ScoringOptions {
    confidence_level: 0.95,
    selection_sharpness: None,
    anchor_index: 0.0,
    selection_cutoff: 0.0005,
    selection_coverage: 1.0,
    target_prior_edges: 5.0,
    regularization_strength: 0.01,
    prior_tau2: 10.0,
    bias_prior_tau2: 2.0,
    bias_prior_logit: 0.0,
}, &judge_info);

for r in &result.rankings {
    println!("Item {}: {:.4} [{:.4}, {:.4}]", r.item, r.score, r.lower_bound, r.upper_bound);
}
```

## Iterative usage with the engine

For iterative ranking (compare, score, choose the next judgements, repeat):

```rust
use nanojudge_core::{
    RankingEngine, EngineConfig, JudgementDistribution,
    Edge, run_scoring, ScoringOptions, JudgeInfo, judge_hash,
    judgements_needed_for_every_item_to_appear_once, calculate_budget,
};

let item_ids: Vec<i64> = vec![10, 20, 30, 40];
let judge_id = judge_hash("my-endpoint", "my-model");
let judge_info = JudgeInfo { judge_ids: vec![judge_id], logprobs_mode: false };

let config = EngineConfig {
    judgement_distribution: JudgementDistribution::TopHeavy,
    matchmaking_sharpness: 1.0,
    min_uniform_edges: 2,
    seed: None, // Some(n) for reproducible pairing
};

let mut engine = RankingEngine::new(&item_ids, config);
let budget = calculate_budget(item_ids.len(), 10, 2); // 10 judgements per item, pairwise
let refit_interval = judgements_needed_for_every_item_to_appear_once(item_ids.len(), 2);
let mut done = 0;

while done < budget {
    // 1. Score existing edges to get posterior summaries for pairing
    if !engine.completed_edges.is_empty() {
        let scoring = run_scoring(&item_ids, &engine.completed_edges, &ScoringOptions {
            confidence_level: 0.95,
            selection_sharpness: Some(0.5),
            anchor_index: 0.0,
            selection_cutoff: 0.0005,
            selection_coverage: 1.0,
            target_prior_edges: 5.0,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            bias_prior_tau2: 2.0,
            bias_prior_logit: 0.0,
        }, &judge_info);
        engine.set_current_posterior(&scoring.item_means, &scoring.item_stds);
        engine.selection_weights = scoring.selection_weights;
    }

    // 2. Engine selects pairs to collect before the next refit
    let judgements_before_refit = refit_interval.min(budget - done);
    let pairs = engine.generate_pairs(judgements_before_refit);

    // 3. You obtain the judgements (call your LLM, ask humans, etc.).
    let results: Vec<Edge> = pairs.iter().map(|&(a, b)| {
        let prob = if a < b { 0.7 } else { 0.3 }; // your LLM call goes here
        Edge::new(a, b, [prob, 1.0 - prob], judge_id)
    }).collect();

    // 4. Feed results back
    engine.record_edges(&results);
    done += results.len();
}
```

## The math

1. **Bradley-Terry MLE** — fast iterative algorithm for point-estimate scores used by lightweight rating updates
2. **Laplace Bradley-Terry inference** — deterministic MAP fitting via Newton-CG, with correlation-aware inverse-Hessian probes producing approximate credible intervals and per-item selection weights in linear work for fixed solver limits
3. **Positional bias estimation** — jointly estimated in the Laplace fit. LLMs tend to favor whichever option is shown first; the model detects and corrects for this automatically
4. **Smart pairing** — selects the next two-item lineups to maximize information gain per judgement

## Modules

| Module | What it does |
|---|---|
| `scoring` | `run_scoring()` — Laplace Bradley-Terry scoring, the main entry point |
| `engine` | `RankingEngine` — iterative orchestrator with smart pair selection |
| `pairing` | Uniform and top-heavy judgement distributions |
| `laplace_bt` | Deterministic MAP fit and matrix-free covariance estimation |
| `bradley_terry` | Fast iterative MLE for quick rating updates between refits |
| `types` | `Edge`, `ScoringOptions`, `ScoringResult`, `RankedItem` |

## Judgement distributions

**Uniform**: Every item gets equal judgement time. Good when you care about the full ranking.

**Top-heavy**: Focuses judgements on items whose standing around the anchor rank is still uncertain. Confidently placed items get the uniform-stage minimum while contested boundary items get many more judgements.

The engine handles two stages automatically:
1. **Uniform stage** (first few refits): uniform pairing until every item has the minimum edge count
2. **Main phase**: your chosen distribution

## Key concepts

- **Win probability**: Not binary win/loss. Each edge carries P(A beats B). A value of 0.73 means "A is probably better but not certain."
- **Ghost player regularization**: A virtual opponent that every item has a tiny draw against. Prevents infinite scores when an item has a 100% or 0% win rate.
- **Positional bias**: Jointly estimated during scoring — no manual calibration needed. The `ScoringResult` reports the estimated bias and its approximate credible interval.

## Design philosophy

Every parameter must be explicitly provided. If required data is missing, the crate panics with a clear message — it never silently falls back to defaults or skips bad input.

Invalid input (duplicate IDs, unknown IDs, mismatched lengths) is treated as a programming error in the caller, not a recoverable condition — there is no error type to handle. Validate at your boundary, then trust the call. Each public function documents its exact panic conditions in a `# Panics` section in the API docs.

## License

MIT
