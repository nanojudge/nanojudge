# nanojudge-core

Pure-computation ranking engine for pairwise comparisons. Part of [nanojudge](https://github.com/nanojudge/nanojudge).

Takes pairwise win probabilities, produces ranked items with confidence intervals. No IO, no HTTP, no filesystem — just math. Bring your own LLM (or any other source of pairwise judgments).

## Quick start

Items are identified by `i64` IDs — any unique numbers you want. The crate handles internal index mapping.

```rust
use nanojudge_core::{run_scoring, ComparisonInput, ScoringOptions, JudgeInfo, stable_hash};

let item_ids = vec![100, 200, 300];
let judge_id = stable_hash("my-endpoint/my-model");

// category_probs = [P(item1 wins), P(item2 wins)]. pairwise() records that
// item1 was shown first and item2 second, so scoring can correct for
// positional bias.
let comparisons = vec![
    ComparisonInput::pairwise(100, 200, [0.8, 0.2], judge_id),
    ComparisonInput::pairwise(200, 300, [0.7, 0.3], judge_id),
];

let judge_info = JudgeInfo { judge_ids: vec![judge_id], logprobs_mode: false };

let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
    iterations: 200,
    burn_in: 100,
    confidence_level: 0.95,
    selection_sharpness: None,
    anchor_index: 0.0,
    selection_cutoff: 0.0005,
    selection_coverage: 1.0,
    target_prior_games: 5.0,
    warm_start: None,
    regularization_strength: 0.01,
    prior_tau2: 10.0,
    proposal_std: 0.3,
    bias_prior_tau2: 2.0,
    bias_proposal_std: 0.15,
    bias_prior_logit: 0.0,
    seed: None,
    inference: nanojudge_core::InferenceMode::Mcmc,
}, &judge_info);

for r in &result.rankings {
    println!("Item {}: {:.4} [{:.4}, {:.4}]", r.item, r.score, r.lower_bound, r.upper_bound);
}
```

## Multi-round usage with the engine

For iterative ranking (compare, score, pick next pairs, repeat):

```rust
use nanojudge_core::{
    RankingEngine, EngineConfig, ComparisonDistribution,
    ComparisonInput, run_scoring, ScoringOptions, JudgeInfo, stable_hash,
};

let item_ids: Vec<i64> = vec![10, 20, 30, 40];
let judge_id = stable_hash("my-endpoint/my-model");
let judge_info = JudgeInfo { judge_ids: vec![judge_id], logprobs_mode: false };

let config = EngineConfig {
    comparison_distribution: ComparisonDistribution::TopHeavy,
    matchmaking_sharpness: 1.0,
    min_uniform_games: 2,
    seed: None, // Some(n) for reproducible pairing
};

let mut engine = RankingEngine::new(&item_ids, config);

for round in 0..20 {
    // 1. Score existing comparisons to get MCMC data for pairing
    if !engine.completed_comparisons.is_empty() {
        let scoring = run_scoring(&item_ids, &engine.completed_comparisons, &ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            // Top-heavy selection: weight each item by its sharpened
            // uncertainty ratio around the anchor (anchor_index 0.0 = the
            // current leader) — items straddling the anchor get the focus.
            // The first item of each pair is drawn from these weights; its
            // opponent comes from info-gain matchmaking in a rating window
            // around it. `None` would disable top-heavy weighting.
            selection_sharpness: Some(0.5),
            anchor_index: 0.0,
            selection_cutoff: 0.0005,
            selection_coverage: 1.0,
            target_prior_games: 5.0,
            warm_start: None,
            regularization_strength: 0.01,
            prior_tau2: 10.0,
            proposal_std: 0.3,
            bias_prior_tau2: 2.0,
            bias_proposal_std: 0.15,
            bias_prior_logit: 0.0,
            seed: None,
            inference: nanojudge_core::InferenceMode::Mcmc,
        }, &judge_info);
        // The posterior (means + stds) drives matchmaking: opponent selection
        // integrates its win probabilities over the rating uncertainty, so
        // matchups that *might* be close also score well.
        engine.set_current_posterior(&scoring.item_means, &scoring.item_stds);
        engine.selection_weights = scoring.selection_weights;
    }

    // 2. Engine decides which pairs to compare
    let pairs = engine.generate_pairs_for_round(round);

    // 3. You perform the comparisons (call your LLM, ask humans, etc.).
    //    The placeholder below stands in for your source of P(a beats b).
    let results: Vec<ComparisonInput> = pairs.iter().map(|&(a, b)| {
        let prob = if a < b { 0.7 } else { 0.3 }; // your LLM call goes here
        ComparisonInput::pairwise(a, b, [prob, 1.0 - prob], judge_id)
    }).collect();

    // 4. Feed results back. No rating refit here: step 1 installs the
    //    posterior before the next pairing. (Standalone `update_current_ratings()`
    //    only matters for flows that pair without an interim scoring pass.)
    engine.record_results(&results);
}
```

## The math

1. **Bradley-Terry MLE** — fast iterative algorithm for point-estimate scores from pairwise win rates
2. **Gaussian BT MCMC** — Bayesian posterior sampling via Metropolis-Hastings within Gibbs, producing confidence intervals and per-item selection weights (each item's uncertainty ratio around the anchor — by default the current leader, configurable via `anchor_index` — so the contested boundary gets the comparisons)
3. **Positional bias estimation** — jointly estimated during MCMC sampling. LLMs tend to favor whichever option is shown first; the sampler detects and corrects for this automatically
4. **Smart pairing** — decides which pairs to compare next to maximize information gain per comparison

## Modules

| Module | What it does |
|---|---|
| `scoring` | `run_scoring()` — unified MCMC wrapper, the main entry point |
| `engine` | `RankingEngine` — multi-round orchestrator with smart pair selection |
| `pairing` | Uniform and top-heavy comparison distributions |
| `gaussian_bt` | Bayesian MCMC sampler (Metropolis-Hastings within Gibbs) |
| `bradley_terry` | Fast iterative MLE for quick rating updates between rounds |
| `types` | `ComparisonInput`, `ScoringOptions`, `ScoringResult`, `RankedItem` |

## Comparison distributions

**Uniform**: Every item gets equal comparison time. Good when you care about the full ranking.

**Top-heavy**: Focuses comparisons on items whose standing around the anchor rank is still uncertain. Confidently placed items (on either side) get the uniform-stage minimum while contested boundary items get many times more. Good for large lists where you mainly care about finding the best items.

The engine handles three stages automatically:
1. **Uniform stage** (first few rounds): uniform pairing until every item has minimum games
2. **Main phase**: your chosen distribution
3. **Smoothing** (last round): reverts to uniform so every item gets fresh data

## Key concepts

- **Win probability**: Not binary win/loss. Each comparison produces P(A beats B) from LLM logprobs. A value of 0.73 means "A is probably better but not certain." This preserves uncertainty through to the final ranking.
- **Warm-start**: Pass `state` from a previous `ScoringResult` back as `warm_start` to skip burn-in. Makes interim scoring fast between rounds.
- **Ghost player regularization**: A virtual opponent that every item has a tiny draw against. Prevents infinite scores when an item has a 100% or 0% win rate.
- **Positional bias**: Jointly estimated during MCMC — no manual calibration needed. The `ScoringResult` reports the estimated bias and its confidence interval.

## Design philosophy

Every parameter must be explicitly provided. If required data is missing, the crate panics with a clear message — it never silently falls back to defaults or skips bad input.

Invalid input (duplicate IDs, unknown IDs, mismatched lengths) is treated as a programming error in the caller, not a recoverable condition — there is no error type to handle. Validate at your boundary, then trust the call. Each public function documents its exact panic conditions in a `# Panics` section in the API docs.

## License

MIT
