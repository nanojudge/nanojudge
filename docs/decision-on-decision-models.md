# Decision on decision models

Decision models (also called "System One" models), such as TypeSafe's Jev, do
not generate text. You give them context and a closed set of answers, and
they return a probability for each answer. They are offered directly by their
labs and through OpenRouter's `decisions` output modality.

NanoJudge does not support them. This document explains why.

## Two items: they work

For a pairwise judgement, a decision model can answer a two-option Choice
question. It returns `{option_1: p, option_2: 1 − p}`, which is exactly the
verdict distribution NanoJudge reads from an LLM's logprobs.

## Three or more items: they don't

A lineup judgement ranks `k` items. With an LLM in logprobs mode, one request
yields `k − 1` distributions. The model writes its ranking one position at a
time, and the logprobs at each step give its choice among the remaining items.

A decision model's Choice question returns only one distribution: who comes
first. Questions in a single request are evaluated independently, so "who
comes second, given the first" cannot be asked at the same time. Getting the
full ranking takes `k − 1` sequential requests, each resending every remaining
item:

| Lineup size | Item texts sent (sequential) | vs one request |
|---|---|---|
| 3 | 3 + 2 = 5 | 1.7× |
| 5 | 5 + 4 + 3 + 2 = 14 | 2.8× |
| 9 | 9 + 8 + … + 2 = 44 | 4.9× |

It also takes `k − 1` round trips per lineup instead of one. At millions of
judgements, that overhead is significant. For lineups, an LLM in logprobs mode
is much more efficient.

## Why not support them for pairs only?

The alternative is a judge type that works only when `lineup_size = 2`. That
would mean maintaining all of this just for pairwise runs:

- a second request path;
- its own config validation, because sampling parameters, deliberation and
  logprobs settings would not apply;
- its own endpoint probe;
- its own failure handling;
- a rule that the judge cannot take part in lineup runs.

A fragmented codebase, where some judges support some features, is harder to
maintain and harder to reason about. Keeping one judge type that supports
everything is the better tradeoff.
