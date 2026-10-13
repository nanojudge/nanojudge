# Reasoning vs deliberation

## What each one is

A judge can think before its verdict in two separate ways:

- **Reasoning** is the model's own thinking. The endpoint returns it
  separately from the answer, and it is often hidden. It is set
  per judge with `reasoning_effort`.
- **Deliberation** is the analysis NanoJudge's prompt asks for, written in the
  visible answer before the verdict. It is set with `deliberation_enabled`
  (or `--deliberation`), and its length with `deliberation_length`.

The **verdict** is the final line, which NanoJudge parses.

### Example

The prompt, from the default deliberation template:

```
Which fruit is healthier?

Option 1:
Strawberry

Option 2:
Pineapple

Instructions:
Write an analysis (2 paragraphs). Analyse both options before forming a preference. You MUST end your response with one of these lines verbatim:

Verdict: Option 1
Verdict: Option 2
```

The output of a reasoning judge with deliberation enabled:

```
[reasoning]
The user wants to know which is healthier. Strawberries: very high vitamin C,
lots of fibre, low sugar. Pineapple: also vitamin C, but more sugar per
serving. Strawberries seem better overall. I'll write the analysis first.

[deliberation]
Strawberries are low in calories and sugar, and are one of the best sources
of vitamin C per serving. They are also rich in fibre and antioxidants.

Pineapple provides vitamin C and manganese, and contains bromelain, an enzyme
that may aid digestion. However, it has noticeably more sugar per serving.

[verdict]
Verdict: Option 1
```

## Why they are separate

**Models differ.** Some models reason and some don't, and some endpoints
require reasoning to be enabled. Deliberation gives a non-reasoning model a way to
think before its verdict. Keeping the two separate lets a reasoning model skip
deliberation instead of thinking twice.

**Deliberation follows instructions.** Deliberation is part of the visible
answer, so it is expected to follow the prompt's instructions, such as its
length and analysing both options before forming a preference, more closely
than reasoning does. This makes it easier to predict total tokens and hence cost.

## The four combinations

Reasoning and deliberation can be combined freely. `max_tokens` in each case:

| Deliberation | Reasoning | `max_tokens` |
|---|---|---|
| On | Off (`reasoning_effort = "none"`) | Optional (default: 2048, or the average of the judges that set it) |
| On | On | Required |
| Off | Off (`reasoning_effort = "none"`) | Optional (default: just enough for the verdict: 16 for pairs, 16 per ranking line for lineups) |
| Off | On | Required |

Reasoning length varies too much between models for any default to fit, so
reasoning judges must always set `max_tokens`.
