# nanojudge

Rank large lists of arbitrary items using LLMs as judges. You provide the items and a comparison criterion (e.g. "Which is healthier?", "Which paper is more impactful?"), and nanojudge runs pairwise comparisons, extracts confidence from token logprobs, and combines everything into a statistically rigorous ranking with confidence intervals.

Works with any OpenAI-compatible API endpoint — local vLLM, OpenAI, Anthropic, etc.

## Install

Download a prebuilt binary from [GitHub Releases](https://github.com/nanojudge/nanojudge/releases), or build from source:

```bash
cargo install --path nanojudge-cli
```

## Usage

```bash
# Rank fruits by healthiness using a local vLLM server
nanojudge rank \
  --criterion "Which fruit is healthier?" \
  --item "Apple" --item "Banana" --item "Mango" --item "Strawberry" \
  --endpoint http://localhost:8000 \
  --model model-id \
  --rounds 10

# Or read items from a file (one per line)
nanojudge rank \
  --criterion "Which movie is more rewatchable?" \
  --items movies.txt \
  --endpoint http://localhost:8000 \
  --model model-id \
  --rounds 20 --concurrency 64

# Pipe items from stdin
cat papers.txt | nanojudge rank \
  --criterion "Which paper is more impactful?" \
  --endpoint http://localhost:8000 \
  --model model-id \
  --rounds 15
```

Output:

```
 # | Item       |   Score | 95% CI Low | 95% CI High | Comparisons
---|------------|---------|------------|-------------|------------
 1 | Mango      |  1.8234 |       1.20 |        2.79 |          18
 2 | Strawberry |  1.0912 |       0.65 |        1.62 |          16
 3 | Apple      |  0.5012 |       0.29 |        0.88 |          17
 4 | Banana     | -0.2103 |      -0.71 |        0.34 |          15

4 items ranked across 10 rounds (30 comparisons)
Position bias — estimated: 0.523 [0.481, 0.567] (corrected for in scores, 0.5 = no bias)
```

Add `--json` for machine-readable output. Add `-v` for progress during execution.

### Saving comparisons for inspection

Save a sample of LLM responses to a JSONL file for spot-checking or live monitoring with `tail -f`:

```bash
# Save all comparisons
nanojudge rank ... --save-comparisons 1.0

# Save ~10% of comparisons
nanojudge rank ... --save-comparisons 0.1

# Save exactly 50 randomly selected comparisons
nanojudge rank ... --save-comparisons 50

# Custom output path (default: comparisons.jsonl)
nanojudge rank ... --save-comparisons 0.3 --save-comparisons-to samples.jsonl
```

Each line is a JSON object with `round`, `item1`, `item2`, `probability`, and `response` (the raw LLM text). Lines are flushed immediately so you can `tail -f` during a run.

## Config file

Save common settings so you don't repeat them every time:

```bash
nanojudge init   # creates ~/.config/nanojudge/config.toml
```

Then set your endpoint, model, and preferred concurrency in the config file. After that you only need:

```bash
nanojudge rank --criterion "Which is better?" --items list.txt
```

CLI flags always override config file values.

## How it works

1. **Pairwise comparisons** — each round, the engine picks which pairs to compare. An LLM judges each pair, and token logprobs give a continuous confidence (not just binary win/loss).

2. **Bradley-Terry scoring** — all pairwise probabilities are combined into global scores using Bayesian MCMC inference (Gaussian Bradley-Terry with Metropolis-Hastings sampling). This produces point estimates plus confidence intervals.

3. **Smart pairing** — the engine decides which pairs to compare next to maximize information gain. Two strategies:
   - **Balanced**: every item gets equal comparison time (good for full rankings)
   - **Top-heavy**: focuses comparisons on top contenders (good for large lists where you mainly want the best items)

4. **Positional bias correction** — LLMs tend to favor whichever option is shown first. The MCMC sampler jointly estimates this bias and corrects for it automatically.

## Workspace structure

This repo is a Cargo workspace with two crates:

| Crate | What it does |
|---|---|
| `nanojudge-core` | Pure-computation ranking engine. No IO — just math. Use this as a Rust dependency. |
| `nanojudge-cli` | Command-line tool that wires the engine to an OpenAI-compatible API. |

## License

MIT
