# Token parser assumptions

## Current assumption

Logprob-based verdict extraction assumes the judge's tokenizer behaves reasonably: the tokens spelling out each verdict (e.g. `"Verdict: A"` and `"Verdict: B"`) tokenize in a consistent, comparable way across outcomes. Under that assumption, parsing a verdict's probability mass is uniform — read the relevant tokens, sum their logprobs, check coverage against `min_logprob_coverage`. Same code path for every outcome.

## The failure mode

A pathological or adversarial tokenizer can tokenize semantically symmetric outcomes asymmetrically. For example `"Verdict: A"` could be a single token while `"Verdict: B"` splits into two. Parsing then stops being uniform:

- Outcome A's probability mass is read in one token; outcome B's requires accumulating across a branching continuation.
- The API's `top_logprobs` only returns a fixed number of candidate tokens per position, so the full continuation tree for the multi-token outcome may not be present at all.
- `min_logprob_coverage` is then computed over incomparable spans — one outcome's coverage is a single read, the other's is a partial walk of a tree that may be truncated.

The extraction logic becomes outcome-specific and branching, rather than a single fixed path.

## What we don't defend against

nanojudge does not harden against adversarial tokenizers. The logprob path assumes that the prompt template's verdict phrasing tokenizes consistently across the outcomes the template emits, and that the returned `top_logprobs` cover enough of each outcome's token span to compute a meaningful coverage. If those assumptions break, logprob verdicts are unreliable.

Every LLM tested so far has a well-behaved tokenizer where verdict outcomes tokenize consistently. This is a theoretical flaw unlikely to be a real issue in practice.
