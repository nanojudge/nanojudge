# Saved judgements

`nanojudge rank` can save every judgement it collects to a JSONL file, one
judgement per line. You can inspect the file, watch it during a run, feed it
into a later run, or re-score it with `nanojudge score`.

## Saving

```bash
# Save successful judgements to judgements-{timestamp}.jsonl in the current directory
nanojudge rank ... --save-successful-judgements

# Save to a specific file
nanojudge rank ... --save-successful-judgements results.jsonl

# Also save failed judgements (unparseable responses) for debugging
nanojudge rank ... --save-successful-judgements --save-failed-judgements

# Include text in successful records (always included in failures)
nanojudge rank ... --save-successful-judgements --save-prompt-text --save-response-text --save-reasoning-text
```

Lines are flushed immediately, so you can follow a run with `tail -f`.

## Record format

Each successful record is a JSON object with these fields:

| Field | Description |
|---|---|
| `refit` | The refit round in which the judgement was collected |
| `item1`, `item2` | The two items, in the order shown to the judge |
| `item1_text_hash`, `item2_text_hash` | SHA-256 hashes (truncated to 64 bits) of the full item text. `nanojudge score` and `--load-judgements` use them as identity keys, and `nanojudge score` rejects records without them |
| `category_probs` | The judge's raw verdict: `[P(item1 wins), P(item2 wins)]`, before any verdict tempering |
| `judge_model`, `judge_endpoint` | The judge that made the judgement |
| `temperature` | The temperature actually sent to the API, after jitter |
| `deliberation` | Whether deliberation was on |
| `reasoning_effort` | The judge's `reasoning_effort` setting, or `null` if unset |
| `criterion` | The criterion the judge was asked about |
| `logprobs` | Whether the run was in logprobs mode |
| `retries_used` | How many retries the judgement needed |
| `hit_max_tokens` | Whether the response was cut off by `max_tokens` |
| `usage` | Token counts, when the endpoint provides them: `prompt_tokens`, `completion_tokens`, `reasoning_tokens` and `visible_tokens`. The last two are `null` if the endpoint doesn't report reasoning tokens |

Text is left out of successful records by default. Each flag adds one field:

| Flag | Field | Contents |
|---|---|---|
| `--save-prompt-text` | `prompt` | What NanoJudge sent |
| `--save-response-text` | `response` | The model's visible answer: deliberation and verdict |
| `--save-reasoning-text` | `reasoning` | The model's own reasoning text |

A text field is `null` when the endpoint sent nothing, and `""` when it sent
an empty string.

Failed records always include `prompt`, `response` and `reasoning`, plus the
same metadata fields. They have no `category_probs`, since no verdict could be
parsed.

### Lineup records

Runs with `lineup_size` above 2 write a different shape, since a lineup has no
fixed number of members:

- `item1`/`item2` are replaced by `items`, an array holding the lineup in
  presentation order.
- `item1_text_hash`/`item2_text_hash` are replaced by `item_text_hashes`.
- `category_probs` is replaced by two fields:
  - `ranking`: the judge's ordering of the lineup, as indices into `items`,
    best first.
  - `place_probs`: one entry per place but the last, giving the probability
    the judge assigned that place's pick among the members not yet placed
    (all `1.0` in text mode).

Pairwise runs are unaffected: a reader written against the two-item shape
keeps working for `lineup_size = 2`.

## Reusing saved judgements

Feed a saved successful-judgements file into a new run to seed its
comparisons before any new ones are collected:

```bash
# Seed this run with judgements saved from an earlier one
nanojudge rank ... --load-judgements results.jsonl
```

The loaded judgements count toward coverage and matchmaking, so new pairings
and the final ranking build on them.

- Loaded records are matched to the run by item text hash, so the file's items
  must be the same as the run's items.
- Every judge that appears in the file must also be a judge in the run.
- A lineup file must match the run's lineup size.
- Mismatches are rejected rather than silently dropped. Edges that reference
  items not in the run are skipped, with a count printed to stderr.
- If the loaded judgements already satisfy `--stop-confidence`, the run
  collects nothing new and scores the loaded judgements directly.

To reuse a judge's saved data without drawing any new comparisons from it,
give that judge `weight = 0` in the config. Its loaded edges still seed the
engine, but it is assigned no new work. At least one judge must have positive
weight.

## Re-scoring with `nanojudge score`

Saved files store raw verdict probabilities; verdict tempering is applied at
scoring time. When re-scoring with `nanojudge score`:

- `--verdict-temperature` sets tempering for all judges. The default is 3.0
  with deliberation and 1.0 without, taken from the file's `deliberation`
  field. Files without a `deliberation` field need `--verdict-temperature`
  passed explicitly.
- `--judge-verdict-temperature "model@endpoint=T"` sets it for one judge.
- `--verdict-temperature 1.0` gives the untempered raw probabilities.

If `rank` used a non-default `verdict_temperature`, globally or per judge,
pass the same values to `score` to reproduce the original ranking. The saved
file doesn't record them.
