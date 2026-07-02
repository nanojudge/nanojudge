# Logprobs problems

## Collapse under reasoning

In non-reasoning mode the judge outputs only a short verdict, and the logprobs over the verdict tokens give a rich distribution — probabilities spanning the full range from 0 to 1 across the outcome categories. This is what makes logprob-based verdict extraction useful: the win probability `p` fed into the Bradley-Terry model is a continuous signal, not a hard pick.

When reasoning is enabled the model typically ends its response with a final paragraph explaining who its winner is. By the time it emits the verdict tokens, it has already committed to a single outcome, and the entire logprob weight collapses onto that one selection. The distribution over outcomes becomes near-degenerate — one category near 1.0, the rest near 0.0.

## Why this wastes reasoning

Any reasoning the model does *after* it has internally settled on a winner adds nothing useful. Once the model has decided who it is going to pick, subsequent tokens are rationalization, not deliberation — the verdict is already fixed and the logprobs at the verdict tokens reflect that commitment rather than genuine uncertainty.

We want the model to reason *before* its verdict, so that the reasoning shapes the verdict. Reasoning after the verdict is decided is wasted compute.

## Keeping the logprob richness

The goal is to keep the rich pre-commitment distribution in the verdict tokens — the model should still be genuinely uncertain at the point it emits the verdict, having reasoned but not prematurely concluded.

One lever is the prompt: instructing the model to keep the verdict hidden until the very end (e.g. "Keep the verdict hidden from the reader until the very end.") discourages it from declaring a winner mid-response and then continuing to "reason" after the fact. How effective this is depends on the particular LLM and how well it obeys such instructions. As LLMs improve at instruction-following, this kind of prompt-level mitigation should become more reliable.

## The "While" death knell

A related, sharper form of premature commitment shows up in sentence structure. Whenever a model begins a sentence "While x is good/great/etc, y...", the item `y` is picked as the winner essentially every time in testing. The phrase "While x" functions as a commitment to `y` before any reasoning is offered.

For example in a comparison between banana and blueberries for healthiness it says the following: "While banana is a healthy choice for cardio based exercises, blueberries..." — there is no need to read further than "While banana" to know with near certainty that blueberries will be the verdict. The contrastive concession has already settled the outcome.

We want models that don't do this. It is hard to get models to recognize when their own sentence structure constitutes a commitment, and harder still to instruct them to avoid it. Adding a long list of phrases and structures to avoid tends to hurt overall model performance — especially for smaller models that degrade when given many constraints at once. So unlike the hidden-verdict lever, this is not cleanly addressable by prompt engineering, and is largely a property of the model being selected rather than something we can fix in the prompt.
