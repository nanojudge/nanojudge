# nanojudge-core

Pure-computation ranking engine for pairwise comparisons. Part of [nanojudge](https://github.com/nanojudge/nanojudge).

Takes pairwise win probabilities, produces ranked items with confidence intervals. No IO, no HTTP, no filesystem — just math. Bring your own LLM (or any other source of pairwise judgments).

## Quick start

Items are identified by `i64` IDs — any unique numbers you want. The crate handles internal index mapping.

```rust
use nanojudge_core::{run_scoring, ComparisonInput, ScoringOptions};

let item_ids = vec![100, 200, 300];

let comparisons = vec![
    ComparisonInput { item1: 100, item2: 200, item1_win_probability: 0.8 },
    ComparisonInput { item1: 200, item2: 300, item1_win_probability: 0.7 },
];

let result = run_scoring(&item_ids, &comparisons, &ScoringOptions {
    iterations: 200,
    burn_in: 100,
    confidence_level: 0.95,
    top_k: 0,
    warm_start: None,
    regularization_strength: 0.01,
});

for r in &result.rankings {
    println!("Item {}: {:.4} [{:.4}, {:.4}]", r.item, r.score, r.lower_bound, r.upper_bound);
}
```

## Multi-round usage with the engine

For iterative ranking (compare, score, pick next pairs, repeat):

```rust
use nanojudge_core::{RankingEngine, EngineConfig, Strategy, ComparisonInput, run_scoring, ScoringOptions};

let item_ids: Vec<i64> = vec![10, 20, 30, 40];
let config = EngineConfig {
    strategy: Strategy::TopHeavy,
    matchmaking_sharpness: 1.0,
    min_games_before_strategy: 3,
    number_of_rounds: Some(20),
};

let mut engine = RankingEngine::new(&item_ids, config);

for round in 0..20 {
    // 1. Score existing comparisons to get MCMC data for pairing
    if !engine.completed_comparisons.is_empty() {
        let scoring = run_scoring(&item_ids, &engine.completed_comparisons, &ScoringOptions {
            iterations: 200,
            burn_in: 100,
            confidence_level: 0.95,
            top_k: 6,
            warm_start: None,
            regularization_strength: 0.01,
        });
        engine.mcmc_top_k_probs = scoring.top_k_probs;
        engine.mcmc_sample_means = scoring.sample_means;
    }

    // 2. Engine decides which pairs to compare
    let pairs = engine.generate_pairs_for_round(round);

    // 3. You perform the comparisons (call your LLM, ask humans, etc.)
    let results: Vec<ComparisonInput> = pairs.iter().map(|&(a, b)| {
        let prob = your_llm_compare(a, b); // you implement this
        ComparisonInput { item1: a, item2: b, item1_win_probability: prob }
    }).collect();

    // 4. Feed results back
    engine.record_results(&results);
    engine.update_current_ratings();
}
```

## The math

1. **Bradley-Terry MLE** — fast iterative algorithm for point-estimate scores from pairwise win rates
2. **Gaussian BT MCMC** — Bayesian posterior sampling via Metropolis-Hastings within Gibbs, producing confidence intervals and P(top K) probabilities
3. **Positional bias estimation** — jointly estimated during MCMC sampling. LLMs tend to favor whichever option is shown first; the sampler detects and corrects for this automatically
4. **Smart pairing** — decides which pairs to compare next to maximize information gain per comparison

## Modules

| Module | What it does |
|---|---|
| `scoring` | `run_scoring()` — unified MCMC wrapper, the main entry point |
| `engine` | `RankingEngine` — multi-round orchestrator with smart pair selection |
| `pairing` | Balanced and top-heavy pairing strategies |
| `gaussian_bt` | Bayesian MCMC sampler (Metropolis-Hastings within Gibbs) |
| `bradley_terry` | Fast iterative MLE for quick rating updates between rounds |
| `types` | `ComparisonInput`, `ScoringOptions`, `ScoringResult`, `RankedItem` |

## Pairing strategies

**Balanced**: Every item gets equal comparison time. Good when you care about the full ranking.

**Top-heavy**: Focuses comparisons on items most likely to be in the top K. Bottom items get the bootstrap minimum while top contenders get 10-50x more. Good for large lists where you mainly care about finding the best items.

The engine handles three stages automatically:
1. **Bootstrap** (first few rounds): balanced pairing until every item has minimum games
2. **Main phase**: your chosen strategy
3. **Smoothing** (last round): reverts to balanced so every item gets fresh data

## Key concepts

- **Win probability**: Not binary win/loss. Each comparison produces P(A beats B) from LLM logprobs. A value of 0.73 means "A is probably better but not certain." This preserves uncertainty through to the final ranking.
- **Warm-start**: Pass `state` from a previous `ScoringResult` back as `warm_start` to skip burn-in. Makes interim scoring fast between rounds.
- **Ghost player regularization**: A virtual opponent that every item has a tiny draw against. Prevents infinite scores when an item has a 100% or 0% win rate.
- **Positional bias**: Jointly estimated during MCMC — no manual calibration needed. The `ScoringResult` reports the estimated bias and its confidence interval.

## Design philosophy

Every parameter must be explicitly provided. If required data is missing, the crate panics with a clear message — it never silently falls back to defaults or skips bad input.

## License

MIT
