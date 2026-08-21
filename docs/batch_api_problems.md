# Batch API problems

OpenAI and Google Batch APIs can take up to 24 hours to return results. That is useful when requests are independent and latency does not matter, but NanoJudge is adaptive: each set of results informs which judgements it requests next.

NanoJudge therefore needs results quickly enough to continue its feedback loop. Waiting up to 24 hours at every refit could turn a normal run into one lasting many days, so real-time endpoints with concurrency are a much better fit.
