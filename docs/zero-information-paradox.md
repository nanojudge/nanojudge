# The zero-information paradox

## Background

The top-heavy selection algorithm depends on the rating of the top item being accurate. `top_mean` — the leader's mean — is the reference point every other item's area is measured against, so it's worth prioritising judgements that inform the strength of the top item.

## The setup

Suppose every judgement run begins with a single-elimination tournament, and for simplicity there are a power-of-2 number of items. After the tournament completes we have a full set of wins and losses, with exactly one item at the top having won all its matches. Surely we now have more information about the strength of the top item?

## The paradox

Not in non-logprobs mode. The shape of the outcome — one undefeated item, one losing only in the final, 2 losing in the semis, and so on — was guaranteed by the structure of the tournament before a single judgement ran. It has probability 1 regardless of the items' true strengths. An event you already knew would happen carries zero information, so conditioning on it cannot move any posterior, including the one over the top item's strength.

Here is the clinching way to see it. Suppose the tournament result *did* tell us something new about the top item's strength. Then — because we knew in advance exactly what shape the result would take — we could have written that conclusion down *before running any judgements at all*. Running the tournament cannot be the source of information we already had a priori. So it supplies none.

The only part of the result that is *not* predetermined is *which* items fill which slots — the identities. That does carry information, but it is purely about *which* item is stronger than which, never about *how much*. Each binary win/loss is consistent with any strength gap: a 51/49 edge and a 99/1 blowout both produce the same recorded win. So the tournament conveys exactly zero information about the particular quantity top-heavy selection depends on: `max(item_strengths)`.

## It is not just the winner

The same argument applies to the runner-up, the third, the fourth — every item near the top, even though all of them were ranked below another item at least once.

Stack it all up and the tournament yields a consistent *ordering* of the top items with *every gap between them undetermined*. The top items are identified and ranked; their strengths are not. You only begin recovering gaps once an item accumulates *both* wins and losses against near-peers, so a win *rate* emerges.

## Why logprobs break the paradox

In logprobs mode each judgement yields a graded probability, not a binary outcome. A 51/49 win tells us the winner is barely stronger; a 99/1 win tells us it's much stronger. The tournament then does inform `max(item_strengths)`, because the margins carry the strength information that binary wins discard. This is one of the concrete ways logprobs mode reduces the number of judgements needed: not just by giving more signal per judgement, but by making otherwise-structure-only judgements actually informative about the quantity the selection algorithm cares about. We would not know before the tournament started what the values of the logprobs would be.
