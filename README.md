# nanojudge

NanoJudge quantifies the relative strengths of arbitrary items under a criterion you define, using LLMs as judges. Provide the criterion (e.g., "Which is healthier?") and your item list of any length (e.g., "Eggs", "Butter", "Spinach", ...), and get a ranking with confidence intervals.

Instead of overwhelming an LLM with one massive prompt, NanoJudge breaks the task down into a series of small lineup judgements. Operating like an intelligent matchmaking league, it adaptively places similarly strong items into lineups as results come in, efficiently producing an accurate leaderboard. The resulting edges are fed into an Elo-style rating system, producing a transparent ranking, all backed by AI explanations.

Works with any OpenAI-compatible API endpoint.

[nanojudge.ai](https://nanojudge.ai) is a hosted version built on this engine, wrapped in a web UI with managed GPU infrastructure.

See the [NanoJudge glossary](docs/glossary.md) for precise definitions of
judgements, edges, refits, coverage, and the adaptive-selection terms used below.

## Install

Download a prebuilt binary from [GitHub Releases](https://github.com/nanojudge/nanojudge/releases), or build from source:

```bash
cargo install --path nanojudge-cli
```

## Usage

First, create a config file with your judge panel:

```bash
nanojudge init   # creates ~/.config/nanojudge/config.toml
```

Example config with multiple judges:

```toml
judgements_per_item = 10
logprobs = true

[[judge]]
endpoint = "http://localhost:8000"
model = "Qwen/Qwen3-4B-Instruct-2507"
weight = 2
temperature = 0.8

[[judge]]
endpoint = "https://api.openai.com/v1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
weight = 3
temperature = 1.0
concurrency = 5
```

Each `[[judge]]` block defines a judge in the panel. Judgements are distributed across judges according to their `weight`. All judges share the same `logprobs` mode.

Then run:

```bash
# Rank items from a file (one per line)
nanojudge rank \
  --criterion "Which fruit is healthier?" \
  --items fruits.txt

# Inline items
nanojudge rank \
  --criterion "Which fruit is healthier?" \
  --item "Guava" --item "Blueberries" --item "Mango" --item "Kiwi"

# Point at a directory — each text file becomes one item
nanojudge rank \
  --criterion "Which essay is more persuasive?" \
  --items essays/

# Pipe items from stdin
cat papers.txt | nanojudge rank \
  --criterion "Which paper is more impactful?"
```

CLI flags like `--judgements-per-item` override config file values.

Output with criterion "Which of these fruits is healthiest?":

```
 # | Item          |   Score | 95% CI Low | 95% CI High | Edges
---|---------------|---------|------------|-------------|------
 1 | guava         |  6.0797 |       5.54 |        6.73 |    13
 2 | raspberries   |  5.3125 |       4.73 |        5.91 |    13
 3 | blueberries   |  5.2954 |       4.72 |        5.85 |    11
 4 | kiwi          |  3.5773 |       3.02 |        4.15 |    13
 5 | pomegranate   |  2.9892 |       2.41 |        3.51 |    13
 6 | passion fruit |  2.4649 |       1.90 |        2.99 |    13
 7 | mango         |  1.2804 |       0.67 |        1.90 |    12
 8 | persimmon     |  0.7152 |       0.17 |        1.21 |    14
 9 | pineapple     | -0.0699 |      -0.67 |        0.50 |    13
10 | figs          | -0.8494 |      -1.42 |       -0.29 |    13
11 | dragon fruit  | -1.2148 |      -1.74 |       -0.65 |    14
12 | tangerines    | -1.4400 |      -1.95 |       -0.93 |    13
13 | bananas       | -1.7869 |      -2.41 |       -1.21 |    12
14 | cherimoya     | -1.9670 |      -2.45 |       -1.43 |    14
15 | watermelon    | -2.0843 |      -2.63 |       -1.54 |    13
16 | durian        | -2.3493 |      -2.91 |       -1.72 |    14
17 | peaches       | -3.2823 |      -3.90 |       -2.65 |    11
18 | lychees       | -3.9726 |      -4.51 |       -3.43 |    14
19 | coconut       | -4.2377 |      -4.77 |       -3.73 |    14
20 | starfruit     | -4.4602 |      -5.00 |       -3.85 |    13
```

Add `--output-format json` for machine-readable output. Add `-v` for progress during execution.

### Saving judgements for inspection

Save judgements to JSONL files for spot-checking or live monitoring with `tail -f`:

```bash
# Save successful judgements to judgements-{timestamp}.jsonl in the current directory
nanojudge rank ... --save-successful-judgements

# Save to a specific file
nanojudge rank ... --save-successful-judgements results.jsonl

# Also save failed judgements (unparseable responses) for debugging
nanojudge rank ... --save-successful-judgements --save-failed-judgements

# Include full prompts and responses in successful records (always included in failures)
nanojudge rank ... --save-successful-judgements --include-successful-prompts
```

Each successful record is a JSON object with `refit`, `item1`, `item2`, `item1_text_hash`, `item2_text_hash`, `category_probs`, `judge_model`, `judge_endpoint`, `temperature` (the actual value sent to the API, after jitter), `verdict_temperature`, `criterion`, `logprobs`, `retries_used`, `hit_max_tokens`, and `usage` (token counts, when the endpoint provides them). The `item*_text_hash` fields are SHA-256 hashes (truncated to 64 bits) of the full item text; `nanojudge score` requires them as identity keys and will reject records without them. Prompts and responses are omitted by default; add `--include-successful-prompts` to include them.

Failed records always include `prompt` and `response` for debugging, plus the same metadata fields.

Lines are flushed immediately so you can `tail -f` during a run.

Runs with `lineup_size` above 2 write a different shape, since a lineup has no fixed number of members: `item1`/`item2` are replaced by an `items` array holding the lineup in presentation order, `item1_text_hash`/`item2_text_hash` by `item_text_hashes`, and `category_probs` by `winner_dist`, the judge's probability that each member of that array won. Pairwise runs are unaffected — a reader written against the two-item shape keeps working for `lineup_size = 2`.

## Config file

The config file lives at `~/.config/nanojudge/config.toml`. Run `nanojudge init` to create one with defaults and documentation for all available options.

Key settings:

| Setting | Description |
|---|---|
| `judgements_per_item` | Average judgements per item. Total budget = `ceil(judgements_per_item * num_items / lineup_size)`. |
| `judgements_per_refit` | Literal number of judgement attempts scheduled between scoring refits, including during uniform pairing. Defaults to enough scheduled judgements for every item to appear at least once. |
| `lineup_size` | Items in each judgement. `2` (default) is the pairwise mode everything else is tuned around; up to `9` is supported, where one call ranks the whole lineup. |
| `logprobs` | `true` to extract logprobs for continuous confidence (requires endpoint support, e.g. vLLM). `false` for text-based verdict parsing (works everywhere, but needs more judgements). |
| `judgement_distribution` | `"uniform"` (default) or `"top-heavy"`. Top-heavy concentrates judgements on the contenders for the top spots. |
| `selection_sharpness` / `cutoff` | Top-heavy tuning. `selection_sharpness` controls how sharply pairing weight concentrates on items near the anchor (lower = more exploration; default `0.7`). `cutoff` drops items below a minimum uncertainty ratio, keeping at least two (`[0,1)`, default `0` = off). |
| `anchor_index` | Which rank anchors top-heavy selection, 0-based best-first (default `0` = leader). `9` = 10th-best, for "find the top ten." Fractional values interpolate between adjacent ranks. |
| `stop_confidence` | Early stop for top-heavy runs: end once the probability that every item is on its correct side of the anchor reaches this value. In `(0.5, 1.0)`, e.g. `0.95`. No default = always use full budget. Top-heavy only. |

Per-judge settings (in `[[judge]]` blocks):

| Setting | Required | Description |
|---|---|---|
| `endpoint` | Yes | OpenAI-compatible API base URL |
| `model` | Yes | Model ID |
| `temperature` | Yes | Sampling temperature |
| `weight` | No | Relative weight for pair assignment (default: 1) |
| `concurrency` | No | Max concurrent requests (default: 16) |
| `max_tokens` | No | Max tokens in response (default: 2048) |
| `api_key_env` | No | Environment variable containing the API key |
| `reasoning_effort` | No | Controls model reasoning mode (e.g. `"none"` to disable Qwen 3.5 thinking) |
| `min_logprob_coverage` | No | Min fraction of verdict-token logprob mass required to trust a verdict, > 0.0 and ≤ 1.0 (default: 0.95) |
| `verdict_temperature` | No | Tempers this judge's parsed verdict distribution before scoring: `q^(1/T)`, dividing each edge's log-odds by T. > 1 softens overconfident verdicts toward 50/50; distinct from the sampling `temperature`. Must be finite and > 0. Also settable top-level (default: 3.0 with reasoning enabled, 1.0 without) |

## How it works

1. **Lineup judgements** — the engine iteratively selects which lineups to present. A lineup is a pair by default; `lineup_size` can widen it to as many as nine items, in which case the judge ranks them all in one call and the ranking is folded back into pairwise edges. Each judge in the panel evaluates its assigned lineups. With `logprobs = true`, token logprobs give continuous confidence. With `logprobs = false`, verdicts are parsed from the response text.

2. **Bradley-Terry scoring** — all edge probabilities are combined into global scores using deterministic Laplace inference. Newton-CG finds the posterior mode, while matrix-free inverse-Hessian probes estimate correlation-aware credible intervals without dense O(n²) matrices.

3. **Adaptive pairing** — the engine uses previous judgements to select the next lineups, maximizing information gain. Two judgement distributions:
   - **Uniform**: every item gets equal judgement time (good for full rankings)
   - **Top-heavy**: focuses judgements on top contenders (good for large lists where you mainly want the best items)

4. **Positional bias correction** — LLMs tend to favor whichever option is shown first. The Bradley-Terry fit jointly estimates this bias and corrects for it automatically.

## Recommended models

NanoJudge works with any instruct-tuned model served over an OpenAI-compatible API. These two score near GPT-5 on [Artificial Analysis](https://artificialanalysis.ai) at a fraction of the price:

| Model | Input per 1M | Output per 1M |
|---|---|---|
| `deepseek/deepseek-v4-flash` | $0.10 | $0.20 |
| `google/gemma-4-31b-it` | $0.12 | $0.37 |

## Workspace structure

This repo is a Cargo workspace with three crates:

| Crate | What it does |
|---|---|
| `nanojudge-core` | Pure-computation ranking engine. No IO — just math. Use this as a Rust dependency. |
| `nanojudge-cli` | Command-line tool that wires the engine to an OpenAI-compatible API. |
| `nanojudge-bench` | Synthetic benchmark harness that measures ranking accuracy against known ground truth. |

## The bigger picture

### A universal engine for subjective sorting

Computer science already has sorting algorithms for numerical data (e.g. QuickSort) and search engines for authority (PageRank) or semantic similarity (Vector Search).

What we've been missing is an engine for subjective criteria - a way to programmatically sort lists by "which is more rewatchable," "which aged the best," or "which code is cleaner."

NanoJudge is **LLM-Sort**. It takes the chaotic, inherently subjective opinions of small, cheap LLMs, runs optimized two-item judgements, and uses Bayesian inference. It is a general-purpose algorithm that turns fuzzy "vibes" into statistically rigorous rankings.

### A coprocessor for agentic systems

This is a fundamental building block for AI architectures. When a large LLM needs to choose between 100+ options, stuffing them into a massive context window is expensive, slow, and prone to "lost in the middle" failures.

Instead, the main LLM can act as the orchestrator: it fetches the candidates, passes the list and the subjective criteria to NanoJudge, uses a much smaller efficient LLM to judge the lineups, and gets back a mathematically grounded ranking.

Every time an AI agent makes a decision, a travel app recommends an itinerary, or a feed ranks content, it is solving a subjective ranking problem. NanoJudge makes that universal process mathematically explicit and can scale to hundreds of thousands of options without running into context length limits.

### Scales with AI progress

NanoJudge is model-agnostic. It uses whatever LLM you point it at as a raw compute engine. That means it surfs the wave of AI progress directly: when a faster, cheaper, smarter LLM releases, changing one config line is all that's needed to make NanoJudge stronger.

## Benchmark

We pitted NanoJudge against **Score-O** — the pointwise scoring baseline from the [BIRCO](https://github.com/BIRCO-benchmark/BIRCO_dataset) paper, on BIRCO's **RELIC** task (recovering a masked quotation in a literary analysis). Both run on the same judge model (Gemma 4 E4B); the only difference is the protocol: NanoJudge compares items pairwise, Score-O scores them one at a time.

Each query in RELIC comes with a pool of passages, one of which is the correct answer and the rest wrong. We made the task harder by generating additional wrong passages with a completion-based LLM, leaving only 1 correct answer as before. 1x is the original dataset. 2x means we doubled the pool size, 4x quadrupled and so on. Score-O collapses as the pool grows whereas NanoJudge degrades gracefully.

| Dataset size | Score-O | NanoJudge |
|---|---|---|
| 1x | 0.3858 | **0.6043** |
| 2x | 0.3158 | **0.5590** |
| 4x | 0.2549 | **0.5325** |
| 8x | 0.1788 | **0.4533** |
| 16x | 0.1546 | **0.3734** |
| 32x | 0.0784 | **0.2958** |

_nDCG@10, higher is better._

At 32x the pool, NanoJudge scores nearly 4x higher than Score-O - and even on a pool 16x larger, it matches Score-O's score on the original, undiluted pool. NanoJudge is built to scale.

## Related work

Qin et al. (2023) showed that pairwise prompting significantly outperforms pointwise and listwise approaches for LLM-based ranking. [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/abs/2306.17563)

## License

MIT
