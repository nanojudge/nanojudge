# Batch API problems

## Background

OpenAI and Google both offer batch APIs: submit a large set of requests at a 50% discount (as of 2026-08-14), and results come back in at most 24 hours. For workloads where every request is known in advance and no request depends on another's result, that is a good trade. nanojudge's comparison loop is not such a workload.

## Pairing depends on previous results

What round N+1 compares is determined by round N's results. In the uniform stage: round 1 pairs items at random (no information exists yet), round 2 pairs rating-adjacent neighbours, and from round 3 onward opponents are drawn from a rating window weighted by information gain. In the top-heavy stage, both *which* items get compared and *which opponents* they get depend on the interim posterior — selection weights concentrate comparisons on the contenders near the anchor, and opponent matchmaking integrates the win probability over each item's current rating uncertainty.

None of this can be precomputed. Every one of these decisions needs a fit that includes all comparisons so far, so every round must wait for the previous round's results. Only round 1 could ever be submitted blind.

## The wall-time arithmetic

Each serial step costs one full batch window — up to 24 hours. A 10-round comparison with one refit per round therefore takes up to 10 days, versus minutes for the same rounds run in real time with concurrency.

It gets worse in normal use. Top-heavy rounds are subdivided into `refits_per_round` chunks, and each chunk runs its own scoring refit and re-derives selection weights before the next chunk's pairs are generated, so each chunk is its own serial step. Worst-case wall time is

    (uniform_rounds + top_heavy_rounds * refits_per_round) * 24 hours

The refits exist to make the pairing more accurate; the more of them a run uses, the more the batch API multiplies the delay.

## Early stopping

`--stop-confidence` ends a run as soon as an interim fit shows the partition confidence has reached the requested level. That only works when you can stop *between* steps: with a batch API the entire comparison budget must be committed up front, so either you over-buy — which eats into the 50% discount — or you under-buy and end with an inconclusive result.

## Submitting all pairs up front does not work

Submitting all N(N-1)/2 pairs in one batch replaces targeted acquisition with an exhaustive round-robin. Most of those pairs are mismatched blowouts whose outcome is near-certain and which therefore carry near-zero information, and the comparison blowup alone cancels the discount before considering that top-heavy concentration and early stopping are both lost. This would be extremely expensive for large lists of items.

## Summary

The batch API trades latency for money. nanojudge trades comparisons for accuracy using a tight feedback loop, and the feedback is the point: every step's inputs depend on the previous step's results. A 50% discount is not worth up to a day of waiting per step.
