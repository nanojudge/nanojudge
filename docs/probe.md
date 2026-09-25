# Endpoint probe

## What it is

Before `nanojudge rank` or `nanojudge benchmark` uses a judge for the first
time, NanoJudge sends the judge's endpoint a few cheap test requests and
records how it behaves. The results are saved to a local file. Later runs use
the saved results instead of probing again. `nanojudge probe` runs a new probe
of every judge in the config on demand.

## Why

NanoJudge relies on endpoints doing what they are asked: returning logprobs
when requested, returning the requested number of top logprobs, not reasoning
when `reasoning_effort = "none"`, and keeping to `max_tokens`. Endpoints don't
always do this, and their documentation is often wrong or silent on these
points.

If NanoJudge simply trusted an endpoint to respect its settings, a mismatch
would only show up partway through a run, as failed judgements, wasted tokens,
or skewed rankings. For example, a model that keeps reasoning with reasoning
turned off can use up `max_tokens` before writing its verdict.

The probe checks the endpoint's behaviour directly with cheap prompts, so
NanoJudge can reject settings the endpoint doesn't honour before a run
starts.

## What it tests

The probe sends four requests, each with the judge's own settings
(temperature, `top_p`, `presence_penalty`, `reasoning_effort` and
`chat_template_kwargs`) and the prompt "What is 55*17? Reply with only the
number.":

- **Reasoning:** a plain request with the judge's `max_tokens`. The model
  reasoned if the reply has reasoning text (`reasoning` or
  `reasoning_content`), reports reasoning tokens, or has `<think>` tags in the
  answer. It also reasoned if the reply used more completion tokens than the
  answer can account for. Every token of the answer is at least one byte, so
  tokens beyond the answer's length in bytes (plus a few special tokens, like
  the end-of-turn token) are reasoning that the endpoint counted but didn't
  show. The prompt asks for a short answer so that hidden reasoning stands out.
- **Logprobs:** a one-token request with logprobs, to check that logprobs come
  back.
- **Top logprobs:** a one-token request for 20 top logprobs, to record how many
  come back.
- **`max_tokens`:** a request limited to 5 tokens, to check the limit is kept.

Each finding is yes, no, or can't tell. The probe can't tell when the endpoint
gives no usable reply, or leaves out the data a finding needs, such as the
token counts.

The probe also records how long each request takes, in milliseconds.

## When runs refuse to start

A run refuses to start, listing every problem it finds, if a judge's probe
shows any of these:

- The endpoint gave no usable reply to the reasoning request.
- The endpoint doesn't keep to `max_tokens`.
- The judge sets `reasoning_effort = "none"`, but the model reasoned.
- The judge uses logprobs, but the endpoint returns none, or returns fewer top
  logprobs than runs ask for (10).

A finding the probe can't make counts against the judge. `rank` only checks
judges with a weight above 0, since judges with weight 0 send no requests.

## Storage

Probe results are saved to `probes.jsonl` in a `nanojudge` folder in the user
data directory (`~/.local/share/nanojudge/` on Linux). The file is
append-only: each probe adds a new line, and old records are never deleted or
overwritten. Each record keeps the raw requests (never headers, so API keys
stay out of the file) and responses along with the findings, so new findings
can later be worked out from old probes without probing again.

A run uses the most recent record whose URL, model, `reasoning_effort`,
`chat_template_kwargs` and `provider` match the judge. Changing one of these settings starts
a new probe; changing others, such as temperature, doesn't. Records from an
older version of the probe are ignored. After fixing an endpoint, run
`nanojudge probe` to replace its saved result.

## Routing endpoints are not supported

Some endpoints route one model name to different backends, which can differ
in logprobs support and reasoning behaviour. A probe of such an endpoint only
describes whichever backend served the probe, not the one that serves the
run. Different requests in the same probe can even be served by different
backends. The probe treats every endpoint as a single, consistent backend, so
its results are not reliable for routing endpoints.

Point each judge at an endpoint that serves the model itself, such as the
model provider's own API or your own server, rather than at a routing
endpoint.

OpenRouter is a routing endpoint, so NanoJudge refuses OpenRouter judges
unless they pin a provider with `provider`, which is sent as OpenRouter's
`provider` request option:

```toml
provider = { only = ["xiaomi"], allow_fallbacks = false }
```

NanoJudge doesn't check what's in `provider`, only that it's set, so make
sure it pins a single provider.
