# Temperature jitter

## What it does

Each judge has a base `temperature`. Temperature jitter applies a per-prompt multiplier drawn from `N(1.0, temperature_jitter)`, clamped to `[0.8, 1.2]`, so every prompt the judge answers gets a slightly different effective temperature. With `temperature_jitter = 0` (the default) the multiplier is always 1.0 and jitter does nothing; raising it spreads the effective temperatures used between refits.

## Why

The goal is stylistic variety in the judge's analyses. We genuinely want a broad range of perspectives on each judgement: testing on real data shows that a panel of judges produces higher-quality rankings than relying on any single judge for all judgements. Jitter is a way to move toward that effect within a single judge — higher temperature variety per prompt produces more variety in the style and emphasis of the analysis, which gets us part of the way toward the broader-range-of-analysis benefit a true panel provides.

It is meant to be used in conjunction with multiple judges, not as a replacement for them. Multiple judges give genuinely different perspectives (different models, different priors); jitter gives one judge a spread of stylistic takes. The two compose.

## An analogy

Imagine generating many 1000×1000 images where each pixel is a random RGB value drawn with mean `(128, 128, 128)` and a fixed standard deviation `(20, 20, 20)`. The pixels are random, but every image is drawn from the same distribution, so they all share a similar style — producing lots of them doesn't give much real variety.

Now instead randomize the standard deviation itself at the start of each image — say `(10, 10, 10)` for one and `(30, 30, 30)` for another. The images are still random, but they now look visibly different from each other, because the spread of the randomness differs.

That difference is what we want. A single judge at a fixed temperature produces analyses that are stylistically similar; jittering the temperature spreads the "style" of the analysis, and that spread is the useful variety.
