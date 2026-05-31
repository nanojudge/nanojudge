# nanojudge

NanoJudge quantifies the relative strengths of arbitrary items under a criterion you define, using LLMs as judges. Provide the criterion (e.g., "Which is healthier?") and your item list of any length (e.g., "Eggs", "Butter", "Spinach", ...), and get a ranking with confidence intervals.

Instead of overwhelming an LLM with one massive prompt, NanoJudge breaks the task down into a series of head-to-head matchups. Operating like an intelligent matchmaking league, it adaptively pairs items of similar strength against each other as the results come in to efficiently produce an accurate leaderboard. These individual wins and losses are fed into an Elo-style rating system, producing a transparent ranking, all backed by AI explanations.

Works with any OpenAI-compatible API endpoint — local vLLM, OpenAI, Anthropic, etc.

[nanojudge.ai](https://nanojudge.ai) is a hosted version built on this engine, wrapped in a web UI with managed GPU infrastructure.

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
rounds = 10
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

Each `[[judge]]` block defines a judge in the panel. Comparisons are distributed across judges according to their `weight`. All judges share the same `logprobs` mode.

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

CLI flags like `--rounds` override config file values.

Output with criterion "Which of these fruits is healthiest?":

```
 # | Item          |   Score | 95% CI Low | 95% CI High | Comparisons | ID
---|---------------|---------|------------|-------------|-------------|----
 1 | guava         |  6.0797 |       5.54 |        6.73 |          13 | 11
 2 | raspberries   |  5.3125 |       4.73 |        5.91 |          13 | 14
 3 | blueberries   |  5.2954 |       4.72 |        5.85 |          11 |  1
 4 | kiwi          |  3.5773 |       3.02 |        4.15 |          13 |  4
 5 | pomegranate   |  2.9892 |       2.41 |        3.51 |          13 |  5
 6 | passion fruit |  2.4649 |       1.90 |        2.99 |          13 |  8
 7 | mango         |  1.2804 |       0.67 |        1.90 |          12 |  0
 8 | persimmon     |  0.7152 |       0.17 |        1.21 |          14 | 15
 9 | pineapple     | -0.0699 |      -0.67 |        0.50 |          13 | 18
10 | figs          | -0.8494 |      -1.42 |       -0.29 |          13 |  7
11 | dragon fruit  | -1.2148 |      -1.74 |       -0.65 |          14 |  9
12 | tangerines    | -1.4400 |      -1.95 |       -0.93 |          13 | 17
13 | bananas       | -1.7869 |      -2.41 |       -1.21 |          12 | 13
14 | cherimoya     | -1.9670 |      -2.45 |       -1.43 |          14 | 19
15 | watermelon    | -2.0843 |      -2.63 |       -1.54 |          13 |  3
16 | durian        | -2.3493 |      -2.91 |       -1.72 |          14 |  2
17 | peaches       | -3.2823 |      -3.90 |       -2.65 |          11 | 10
18 | lychees       | -3.9726 |      -4.51 |       -3.43 |          14 |  6
19 | coconut       | -4.2377 |      -4.77 |       -3.73 |          14 | 16
20 | starfruit     | -4.4602 |      -5.00 |       -3.85 |          13 | 12
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

The config file lives at `~/.config/nanojudge/config.toml`. Run `nanojudge init` to create one with defaults and documentation for all available options.

Key settings:

| Setting | Description |
|---|---|
| `rounds` | Number of comparison rounds |
| `logprobs` | `true` to extract logprobs for continuous confidence (requires endpoint support, e.g. vLLM). `false` for text-based verdict parsing (works everywhere, but needs more rounds). |
| `strategy` | `"balanced"` (default) or `"top-heavy"` |

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

## How it works

1. **Pairwise comparisons** — each round, the engine picks which pairs to compare. Each judge in the panel evaluates its assigned pairs. With `logprobs = true`, token logprobs give continuous confidence. With `logprobs = false`, verdicts are parsed from the response text.

2. **Bradley-Terry scoring** — all pairwise probabilities are combined into global scores using Bayesian MCMC inference (Gaussian Bradley-Terry with Metropolis-Hastings sampling). This produces point estimates plus confidence intervals.

3. **Adaptive pairing** — the engine uses the results of previous comparisons to decide what to compare next, maximizing information gain. Two strategies:
   - **Balanced**: every item gets equal comparison time (good for full rankings)
   - **Top-heavy**: focuses comparisons on top contenders (good for large lists where you mainly want the best items)

4. **Positional bias correction** — LLMs tend to favor whichever option is shown first. The MCMC sampler jointly estimates this bias and corrects for it automatically.

## Recommended models

NanoJudge works with any instruct-tuned model served over an OpenAI-compatible API. These two score near GPT-5 on [Artificial Analysis](https://artificialanalysis.ai) at a fraction of the price:

| Model | Input per 1M | Output per 1M |
|---|---|---|
| `deepseek/deepseek-v4-flash` | $0.10 | $0.20 |
| `google/gemma-4-31b-it` | $0.12 | $0.37 |

## Workspace structure

This repo is a Cargo workspace with two crates:

| Crate | What it does |
|---|---|
| `nanojudge-core` | Pure-computation ranking engine. No IO — just math. Use this as a Rust dependency. |
| `nanojudge-cli` | Command-line tool that wires the engine to an OpenAI-compatible API. |

## The bigger picture

### A universal engine for subjective sorting

Computer science already has sorting algorithms for numerical data (e.g. QuickSort) and search engines for authority (PageRank) or semantic similarity (Vector Search).

What we've been missing is an engine for subjective criteria - a way to programmatically sort lists by "which is more rewatchable," "which aged the best," or "which code is cleaner."

Nanojudge is **LLM-Sort**. It takes the chaotic, inherently subjective opinions of small, cheap LLMs, runs optimized pairwise comparisons, and uses Bayesian inference. It is a general-purpose algorithm that turns fuzzy "vibes" into statistically rigorous rankings.

### A coprocessor for agentic systems

This is a fundamental building block for AI architectures. When a large LLM needs to choose between 100+ options, stuffing them into a massive context window is expensive, slow, and prone to "lost in the middle" failures.

Instead, the main LLM can act as the orchestrator: it fetches the candidates, passes the list and the subjective criteria to nanojudge, uses a much smaller efficient LLM to do the pairwise comparisons, and gets back a mathematically grounded ranking.

Every time an AI agent makes a decision, a travel app recommends an itinerary, or a feed ranks content, it is solving a subjective ranking problem. Nanojudge makes that universal process mathematically explicit and can scale to hundreds of thousands of options without running into context length limits.

### Scales with AI progress

Nanojudge is model-agnostic. It uses whatever LLM you point it at as a raw compute engine. That means it surfs the wave of AI progress directly: as small, open-weight models get faster, cheaper, and smarter, Nanojudge inherits every one of those gains and becomes a more powerful engine.

## Related work

Qin et al. (2023) showed that pairwise prompting significantly outperforms pointwise and listwise approaches for LLM-based ranking. [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/abs/2306.17563)

## License

MIT
