# NanoJudge glossary

## Core units

- **Item**: One candidate being ranked, such as one answer, product, document,
  or model output.

- **Criterion**: The question used to decide which item is better, such as
  “Which explanation is more accurate?”

- **Lineup**: The items shown together in one judgement. `lineup_size = 2` is a
  pair; larger lineups contain 3 to 9 items.

- **Judgement**: One requested evaluation of one lineup by one judge. Retries
  belong to that same judgement attempt. A failed or unparseable attempt
  consumes budget but contributes no scoring evidence.

- **Winner distribution**: For a lineup, the judge's probability that each
  member is the best item. In saved lineup judgements this is `winner_dist`.

- **Category probabilities** (`category_probs`): An edge's two probabilities,
  `[P(item1 wins), P(item2 wins)]`.

- **Edge**: One pairwise scoring observation: two items plus the probability
  that the first beats the second. Bradley–Terry scoring consumes edges, not raw
  judgements.

  A two-item judgement produces one edge. A successful lineup judgement can
  produce up to `lineup_size * (lineup_size - 1) / 2` edges. For example, one
  three-item judgement can produce three edges, and each item is then incident
  to two of them. A pair whose two winner probabilities are both zero produces
  no edge.

- **Edge count**: The number of surviving derived edges involving an item. It is
  not the number of judgements containing that item. Uniform-stage thresholds
  and top-heavy coverage use edge counts.

- **Edge weight**: How much an edge contributes to the scoring likelihood. A
  pair judgement has weight 1. Edges derived from one lineup share that
  judgement's available information, so they do not each count as an
  independent full judgement.

- **Win probability**: The probability carried by an edge that its first item
  beats its second. It may be continuous, such as 0.73, rather than a binary win
  or loss.

## Run structure

- **Judgement budget**: The maximum number of judgement attempts scheduled for
  a run. It is calculated from the item count, `judgements_per_item`, and
  `lineup_size`.

- **Judgements per item** (`judgements_per_item`): The average item-appearance
  target used to calculate the judgement budget:
  `ceil(judgements_per_item * num_items / lineup_size)`.

- **Refit**: Re-running the scoring model on all successful edges collected so
  far. The updated scores and uncertainties then inform subsequent matchmaking.

- **Judgements per refit** (`judgements_per_refit`): The number of judgement
  attempts scheduled between refits. Smaller values adapt sooner; larger values
  make fewer scoring passes. It does not change the total budget. The default is
  enough scheduled judgements for every item to appear at least once.

- **Interim ranking**: A ranking produced at a refit before the final judgement
  budget has been exhausted.

## Selection and matchmaking

- **Judgement distribution**: The policy that allocates judgement opportunities
  among items. This is either uniform or top-heavy. It is unrelated to the
  probability distribution returned by a judge.

- **Uniform**: Does not use top-heavy selection weights. It keeps evidence broad
  by prioritising the items with the lowest cumulative edge counts. “Uniform”
  describes item allocation; opponents can still be selected intelligently
  using ratings and information gain.

- **Uniform stage**: A period in which NanoJudge uses uniform item allocation.
  In a uniform run this lasts for the entire run. In a top-heavy run it is the
  initial stage, lasting until every item has at least `min_uniform_edges`
  incident edges. The transition in a top-heavy run is driven by edge counts,
  not by how many judgements have been scheduled or how often every item has
  appeared. `min_uniform_edges` is therefore an edge threshold.

- **Top-heavy**: Allocates more judgements to items whose positions around the
  chosen anchor are unresolved. It first selects a focal item from the selection
  weights, then selects informative opponents for it.

- **Focal item**: The first item chosen for a top-heavy judgement. Its selection
  weight controls how likely it is to be chosen.

- **Opponent**: An additional item placed with the focal item. Pair judgements
  have one opponent; larger lineups have several distinct opponents.

- **Matchmaking**: Choosing which items should be judged together. During
  uniform allocation NanoJudge starts with random matching when some items have
  no edges, uses nearest-rating matching when the minimum edge count is one, and
  uses information-gain matching once the minimum is at least two.

- **Information gain**: An estimate of how useful a possible matchup will be for
  separating the items' strengths. Matchups that may be close are generally more
  informative than obvious mismatches.

- **Matchmaking sharpness** (`matchmaking_sharpness`): Controls how strongly
  opponent selection favours high-information matchups. It affects opponent
  choice, not which item becomes the focal item.

- **Selection weight**: A top-heavy item's relative chance of becoming the focal
  item. It is based on uncertainty around the anchor, then adjusted by selection
  sharpness, cutoff, and top-heavy coverage.

- **Anchor-uncertainty ratio**: The smaller of an item's probabilities of being
  above or below the anchor, divided by the larger. It is 1 when the item is
  maximally unresolved around the anchor and approaches 0 when the item's side
  is clear. This is the base of its top-heavy selection weight.

- **Selection sharpness** (`selection_sharpness`): Raises the anchor-uncertainty
  ratio to a power. Lower values flatten focal-item probabilities and explore
  more; higher values concentrate them more strongly.

- **Selection cutoff** (`cutoff`): Removes items whose anchor-uncertainty ratio
  is below the threshold. The two highest-ratio candidates are always retained
  so a valid judgement can still be formed.

- **Anchor**: The rank boundary that top-heavy selection tries to resolve.
  `anchor_index = 0` targets the current best item; `anchor_index = 9` targets
  the boundary around tenth place. Fractional indices target the space between
  adjacent ranks.

- **Selection target**: The strength value against which anchor uncertainty is
  measured. Early in a run it can blend the observed anchor strength with a
  prior prediction; as the anchor gains edges, it converges to the observed
  anchor.

- **Target prior edges** (`target_prior_edges`): The pseudo-edge strength of the
  prior-predicted selection target. Zero disables the blend; larger values make
  the prior prediction fade more slowly.

## Coverage settings

- **Top-heavy coverage** (`coverage`): A proportional-fair adjustment to focal
  selection. An item's base selection weight is divided by its edge count raised
  to this value. `0` disables the adjustment; `1` is the standard proportional-
  fair setting. This does not choose the opponent—the information-gain
  matchmaker does that afterward.

- **Minimum logprob coverage** (`min_logprob_coverage`): The minimum fraction of
  verdict-token probability mass that must be present in the API's returned top
  logprobs before NanoJudge trusts the parsed verdict. This is a parser-quality
  threshold and has nothing to do with item or edge allocation.

## Judges and verdicts

- **Judge**: One endpoint-and-model combination that evaluates lineups.

- **Judge panel**: All configured judges used in a run. NanoJudge estimates
  positional bias separately for each judge.

- **Judge weight**: A judge's relative share of scheduled requests. This is
  unrelated to item selection weights and edge weights.

- **Logprobs mode**: Uses token probabilities to recover a continuous verdict
  distribution. Text mode instead parses the stated winner and produces a
  one-hot verdict.

- **Sampling temperature** (`temperature`): Controls randomness while the LLM
  generates its response.

- **Verdict temperature** (`verdict_temperature`): Softens or sharpens the
  already-parsed verdict probabilities before scoring. It does not change LLM
  generation.

- **Position or slot bias**: A judge's tendency to favour an item because of
  where it appeared in the prompt. NanoJudge estimates this during scoring and
  corrects for it.

## Scoring and uncertainty

- **Bradley–Terry model**: The pairwise model that converts all edge win
  probabilities into one globally consistent strength for each item.

- **MLE rating**: A quick maximum-likelihood point estimate of item strength.
  NanoJudge can use these provisional ratings for matchmaking when a full
  uncertainty fit is unnecessary.

- **MAP / Laplace fit**: The full scoring fit. MAP is the most probable set of
  strengths after combining the edges with the priors; the Laplace
  approximation uses the curvature around that point to estimate uncertainty.

- **Score / log-strength**: The fitted strength reported for an item. Scores are
  mean-centred, so their ordering and differences matter; the absolute zero does
  not.

- **Posterior mean and standard deviation**: The current strength estimate and
  its uncertainty. Matchmaking uses both, so an uncertain matchup can remain
  valuable even when its current mean scores are not close.

- **Confidence interval / credible interval**: The reported approximate
  posterior range around an item's score at the configured confidence level.
  The CLI labels it `CI`; internal documentation may call it credible.

- **Ghost player**: A weak virtual opponent used for regularization. It keeps
  scores finite when an item has only wins or only losses.

- **Regularization strength** (`regularization_strength`): Controls the ghost
  player's influence. Stronger regularization pulls extreme scores inward more.

- **Prior tau-squared** (`prior_tau2`): The prior variance of item log-strengths.
  Larger values allow strengths to spread farther before evidence pulls them
  together.

- **Bias prior tau-squared** (`bias_prior_tau2`): The prior variance of judge
  position bias in logit space.

- **Stop confidence** (`stop_confidence`): A top-heavy early-stopping threshold
  for the product of the checked items' probabilities of being on their
  posterior-favoured side of the observed anchor. It is separate from the
  confidence level used for reported score intervals.

## Synthetic benchmark terms

- **Trial**: One independently generated synthetic ranking problem and complete
  NanoJudge run.

- **Actual tau-squared** (`actual_tau2`): The variance used to generate the
  synthetic items' true latent strengths.

- **Spearman rho**: Rank correlation between the inferred and true complete
  rankings. Higher is better; 1 is perfect.

- **Top-k displacement**: The mean absolute difference between inferred and true
  positions for the true top-k items. Lower is better; 0 is perfect.
