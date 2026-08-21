/// Fake OpenAI-compatible endpoint backed by a Bradley-Terry strength table.
///
/// Receives chat completion requests, extracts the item IDs from the prompt,
/// looks up their true strengths, and estimates each verdict-token distribution
/// with repeated Bradley-Terry/Plackett-Luce samples. The empirical distribution
/// is returned in the OpenAI logprobs response shape that the real NanoJudge CLI
/// expects, while the whole request still counts as one judgement.
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Request types (subset of OpenAI chat completion API)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<RequestMessage>,
    #[serde(default)]
    logprobs: Option<bool>,
}

#[derive(Deserialize)]
struct RequestMessage {
    content: String,
}

// ---------------------------------------------------------------------------
// Response types (matching OpenAI chat completion API)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: ResponseUsage,
}

#[derive(Serialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChoiceLogprobs>,
    finish_reason: String,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChoiceLogprobs {
    content: Vec<LogprobToken>,
}

#[derive(Serialize)]
struct LogprobToken {
    token: String,
    top_logprobs: Option<Vec<TopLogprobEntry>>,
}

#[derive(Serialize)]
struct TopLogprobEntry {
    token: String,
    logprob: f64,
}

#[derive(Serialize)]
struct ResponseUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

pub struct JudgeState {
    /// Secret strength table: item text -> true strength.
    pub strengths: HashMap<String, f64>,
    /// Base seed for deterministic per-pair verdict derivation.
    pub seed: u64,
    /// Independent draws used to estimate each verdict-token distribution.
    pub samples_per_judgement: usize,
    /// Per-pair encounter counter so repeated matchups get independent sample sets.
    pub encounter_counts: Mutex<HashMap<(String, String), u64>>,
}

// ---------------------------------------------------------------------------
// Prompt parsing
// ---------------------------------------------------------------------------

/// Extract the two item texts from a NanoJudge judgement prompt.
///
/// Relies on the fixed structure of the default prompt template:
///   Option 1:\n<item1>\n\nOption 2:\n<item2>\n\nInstructions:
fn extract_items(prompt: &str) -> (String, String) {
    let opt1_marker = "Option 1:\n";
    let opt2_marker = "Option 2:\n";

    let opt1_start = prompt
        .find(opt1_marker)
        .expect("Prompt missing 'Option 1:' marker")
        + opt1_marker.len();
    let opt2_pos = prompt
        .find(opt2_marker)
        .expect("Prompt missing 'Option 2:' marker");
    let item1 = prompt[opt1_start..opt2_pos].trim().to_string();

    let opt2_start = opt2_pos + opt2_marker.len();
    let opt2_end = prompt[opt2_start..]
        .find("\n\nInstructions:")
        .map(|p| opt2_start + p)
        .unwrap_or(prompt.len());
    let item2 = prompt[opt2_start..opt2_end].trim().to_string();

    (item1, item2)
}

/// True if this is a lineup judgement prompt (3+ items, labelled by letter).
fn is_lineup_judgement(prompt: &str) -> bool {
    prompt.contains("Option A:\n")
}

/// Option letters a lineup prompt can use, in order.
const RANK_LETTERS: [char; 9] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];

/// Extract the item texts from a lineup prompt, in slot order. Mirrors
/// `extract_items` but for the "Option A:/Option B:/..." layout, at whatever
/// lineup size the prompt was built for.
fn extract_lineup_items(prompt: &str) -> Vec<String> {
    // Marker positions, in slot order, for as many options as the prompt has.
    let mut marker_positions: Vec<(usize, usize)> = Vec::new();
    for letter in RANK_LETTERS {
        let marker = format!("Option {letter}:\n");
        match prompt.find(&marker) {
            Some(pos) => marker_positions.push((pos, marker.len())),
            // Options are contiguous from A, so the first gap ends the lineup.
            None => break,
        }
    }
    assert!(
        marker_positions.len() >= 2,
        "lineup prompt must contain at least two 'Option <letter>:' markers"
    );

    let instructions = prompt
        .find("\n\nInstructions:")
        .unwrap_or(prompt.len());

    marker_positions
        .iter()
        .enumerate()
        .map(|(i, &(pos, marker_len))| {
            let start = pos + marker_len;
            let end = marker_positions
                .get(i + 1)
                .map(|&(next_pos, _)| next_pos)
                .unwrap_or(instructions);
            prompt[start..end].trim().to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Verdict generation
// ---------------------------------------------------------------------------

/// One emitted verdict token plus the empirical distribution estimated from a
/// set of independent draws. The emitted token is the first draw in the
/// set, matching normal generation while guaranteeing it has nonzero mass in
/// the empirical distribution.
#[derive(Debug)]
struct SampledToken<const N: usize> {
    emitted: usize,
    probs: [f64; N],
}

/// Sample an index from a finite categorical distribution. The probabilities
/// need not sum to exactly one; they are treated as nonnegative weights.
fn sample_categorical<const N: usize>(probs: [f64; N], rng: &mut impl Rng) -> usize {
    let total: f64 = probs.iter().sum();
    assert!(
        total.is_finite() && total > 0.0 && probs.iter().all(|p| p.is_finite() && *p >= 0.0),
        "categorical probabilities must be finite, nonnegative, and have positive mass"
    );

    let mut draw = rng.random::<f64>() * total;
    for (idx, &prob) in probs.iter().enumerate() {
        if draw < prob {
            return idx;
        }
        draw -= prob;
    }

    // Floating-point subtraction can leave a tiny positive remainder. Return
    // the final positive-mass category rather than inventing a different draw.
    probs.iter().rposition(|&p| p > 0.0).unwrap()
}

/// Estimate one token distribution with `samples` independent categorical
/// draws. Every output probability is an exact multiple of `1 / samples`.
fn sample_token_distribution<const N: usize>(
    probs: [f64; N],
    samples: usize,
    rng: &mut impl Rng,
) -> SampledToken<N> {
    assert!(samples > 0, "samples_per_judgement must be at least 1");

    let mut counts = [0usize; N];
    let mut emitted = 0;
    for sample_idx in 0..samples {
        let outcome = sample_categorical(probs, rng);
        if sample_idx == 0 {
            emitted = outcome;
        }
        counts[outcome] += 1;
    }

    let denominator = samples as f64;
    let probs = std::array::from_fn(|i| counts[i] as f64 / denominator);
    SampledToken { emitted, probs }
}

/// Derive a deterministic seed from (base_seed, every item in the lineup,
/// encounter).
///
/// Every item is hashed, in prompt order, so that neither a positional swap nor
/// a change to any one member of the lineup reuses another lineup's RNG stream.
/// The encounter counter ensures repeated matchups of the same ordered lineup
/// get independent sample sets.
fn deterministic_lineup_seed(base: u64, items: &[&str], seq: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    base.hash(&mut hasher);
    for item in items {
        item.hash(&mut hasher);
    }
    seq.hash(&mut hasher);
    hasher.finish()
}

fn verdict_text(winner: usize) -> &'static str {
    match winner {
        0 => "Verdict: Option 1",
        1 => "Verdict: Option 2",
        _ => unreachable!(),
    }
}

/// Build a pairwise logprobs payload from an empirical verdict distribution.
/// Zero-count alternatives are omitted, just as a real top-logprobs list may
/// omit tokens outside its returned mass; the parser therefore preserves exact
/// empirical zeros without requiring a finite stand-in for ln(0).
fn build_logprobs_payload(winner: usize, probs: [f64; 2]) -> ChoiceLogprobs {
    let top_logprobs: Vec<TopLogprobEntry> = ["1", "2"]
        .iter()
        .enumerate()
        .filter(|(i, _)| probs[*i] > 0.0)
        .map(|(i, d)| TopLogprobEntry {
            token: d.to_string(),
            logprob: probs[i].ln(),
        })
        .collect();

    ChoiceLogprobs {
        content: vec![
            LogprobToken {
                token: "Verdict".to_string(),
                top_logprobs: None,
            },
            LogprobToken {
                token: ":".to_string(),
                top_logprobs: None,
            },
            LogprobToken {
                token: " Option".to_string(),
                top_logprobs: None,
            },
            LogprobToken {
                token: format!(" {}", winner + 1),
                top_logprobs: Some(top_logprobs),
            },
        ],
    }
}

/// Softmax over a lineup's strengths, giving the Plackett-Luce top-1
/// probabilities.
fn softmax(vals: &[f64]) -> Vec<f64> {
    let m = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = vals.iter().map(|v| (v - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// Estimate one token distribution with `samples` independent categorical
/// draws, over a distribution whose width is known only at runtime. The slice
/// analogue of `sample_token_distribution`; the emitted token is the first draw.
fn sample_token_distribution_dyn(
    probs: &[f64],
    samples: usize,
    rng: &mut impl Rng,
) -> (usize, Vec<f64>) {
    assert!(samples > 0, "samples_per_judgement must be at least 1");

    let mut counts = vec![0usize; probs.len()];
    let mut emitted = 0;
    for sample_idx in 0..samples {
        let outcome = sample_categorical_dyn(probs, rng);
        if sample_idx == 0 {
            emitted = outcome;
        }
        counts[outcome] += 1;
    }

    let denominator = samples as f64;
    let probs = counts.iter().map(|&c| c as f64 / denominator).collect();
    (emitted, probs)
}

/// `sample_categorical` over a runtime-width distribution.
fn sample_categorical_dyn(probs: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = probs.iter().sum();
    assert!(
        total.is_finite() && total > 0.0 && probs.iter().all(|p| p.is_finite() && *p >= 0.0),
        "categorical probabilities must be finite, nonnegative, and have positive mass"
    );

    let mut draw = rng.random::<f64>() * total;
    for (idx, &prob) in probs.iter().enumerate() {
        if draw < prob {
            return idx;
        }
        draw -= prob;
    }
    probs.iter().rposition(|&p| p > 0.0).unwrap()
}

/// Estimate the autoregressive token distributions of a lineup ranking.
///
/// Sequential Plackett-Luce: first place gets `samples` draws over every item,
/// and its first draw is the emitted winner. Conditional on that emitted token,
/// second place gets a fresh `samples` draws over the items still unplaced, and
/// so on down the ranking. The last place is deterministic — one item remains.
///
/// Returns the emitted ranking (item indices, best first) and one distribution
/// per rank position, each over all items (zero on the already-placed ones).
fn sample_lineup_ranking(
    strengths: &[f64],
    samples: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let size = strengths.len();
    let mut remaining: Vec<usize> = (0..size).collect();
    let mut order: Vec<usize> = Vec::with_capacity(size);
    let mut dists: Vec<Vec<f64>> = Vec::with_capacity(size);

    while remaining.len() > 1 {
        let local_strengths: Vec<f64> = remaining.iter().map(|&i| strengths[i]).collect();
        let (emitted_local, local_probs) =
            sample_token_distribution_dyn(&softmax(&local_strengths), samples, rng);

        // Lift the local distribution back onto all `size` options.
        let mut dist = vec![0.0_f64; size];
        for (local_idx, &item) in remaining.iter().enumerate() {
            dist[item] = local_probs[local_idx];
        }
        dists.push(dist);

        let emitted = remaining[emitted_local];
        order.push(emitted);
        remaining.retain(|&i| i != emitted);
    }

    // Last place: the one item left, with probability 1.
    let last = remaining[0];
    let mut dist = vec![0.0_f64; size];
    dist[last] = 1.0;
    dists.push(dist);
    order.push(last);

    (order, dists)
}

/// Ordinal words for the ranking lines, matching the CLI's lineup template.
const RANK_ORDINALS: [&str; 9] = [
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
];

/// Render the ranking as the "Nth place is Option X" lines the default lineup
/// template asks for.
fn lineup_judgement_text(order: &[usize]) -> String {
    order
        .iter()
        .enumerate()
        .map(|(rank, &item)| format!("{} place is Option {}", RANK_ORDINALS[rank], RANK_LETTERS[item]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the logprobs payload for a lineup ranking. Each "Nth place is Option"
/// line ends in a letter token carrying that rank's empirical distribution over
/// the option letters. "Option" is the parser's anchor.
fn build_lineup_logprobs(order: &[usize], dists: &[Vec<f64>]) -> ChoiceLogprobs {
    let mut content = Vec::new();
    for (rank, &item) in order.iter().enumerate() {
        let dist = &dists[rank];
        let top: Vec<TopLogprobEntry> = (0..dist.len())
            .filter(|&i| dist[i] > 0.0)
            .map(|i| TopLogprobEntry {
                token: RANK_LETTERS[i].to_string(),
                logprob: dist[i].ln(),
            })
            .collect();
        content.extend(vec![
            LogprobToken { token: RANK_ORDINALS[rank].to_string(), top_logprobs: None },
            LogprobToken { token: " place".to_string(), top_logprobs: None },
            LogprobToken { token: " is".to_string(), top_logprobs: None },
            LogprobToken { token: " Option".to_string(), top_logprobs: None },
            LogprobToken { token: format!(" {}", RANK_LETTERS[item]), top_logprobs: Some(top) },
            LogprobToken { token: "\n".to_string(), top_logprobs: None },
        ]);
    }
    ChoiceLogprobs { content }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn handle_chat(
    State(state): State<Arc<JudgeState>>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let prompt = &request.messages.last().expect("empty messages array").content;
    let want_logprobs = request.logprobs.unwrap_or(false);

    if is_lineup_judgement(prompt) {
        return handle_lineup_judgement(state, prompt, want_logprobs);
    }

    let (item1, item2) = extract_items(prompt);

    let s1 = *state
        .strengths
        .get(&item1)
        .unwrap_or_else(|| panic!("unknown item: {item1:?}"));
    let s2 = *state
        .strengths
        .get(&item2)
        .unwrap_or_else(|| panic!("unknown item: {item2:?}"));

    let sampled_verdict = {
        let key = (item1.clone(), item2.clone());
        let encounter = {
            let mut counts = state.encounter_counts.lock().unwrap();
            let n = counts.entry(key).or_insert(0);
            let current = *n;
            *n += 1;
            current
        };
        let pair_seed = deterministic_lineup_seed(state.seed, &[&item1, &item2], encounter);
        let mut rng = StdRng::seed_from_u64(pair_seed);
        let p1 = 1.0 / (1.0 + (-(s1 - s2)).exp());
        sample_token_distribution(
            [p1, 1.0 - p1],
            state.samples_per_judgement,
            &mut rng,
        )
    };

    Json(ChatResponse {
        choices: vec![Choice {
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: verdict_text(sampled_verdict.emitted).to_string(),
            },
            logprobs: if want_logprobs {
                Some(build_logprobs_payload(
                    sampled_verdict.emitted,
                    sampled_verdict.probs,
                ))
            } else {
                None
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ResponseUsage {
            prompt_tokens: 50,
            completion_tokens: 5,
        },
    })
}

/// Handle a lineup judgement request: sample a full ranking as a sequential
/// Plackett-Luce chain, estimating each rank's token distribution from repeated
/// draws over the items still unplaced.
fn handle_lineup_judgement(
    state: Arc<JudgeState>,
    prompt: &str,
    want_logprobs: bool,
) -> Json<ChatResponse> {
    let items = extract_lineup_items(prompt);
    let strengths: Vec<f64> = items
        .iter()
        .map(|item| {
            *state
                .strengths
                .get(item)
                .unwrap_or_else(|| panic!("unknown item: {item:?}"))
        })
        .collect();

    // Independent draw per repeated (ordered) lineup, keyed via the shared
    // encounter counter (the second key slot is unused for lineups).
    let key = (items.join("\u{0}"), String::new());
    let encounter = {
        let mut counts = state.encounter_counts.lock().unwrap();
        let n = counts.entry(key).or_insert(0);
        let current = *n;
        *n += 1;
        current
    };
    let item_refs: Vec<&str> = items.iter().map(String::as_str).collect();
    let seed = deterministic_lineup_seed(state.seed, &item_refs, encounter);
    let mut rng = StdRng::seed_from_u64(seed);

    let (order, dists) = sample_lineup_ranking(&strengths, state.samples_per_judgement, &mut rng);
    let content = lineup_judgement_text(&order);
    let logprobs = if want_logprobs {
        Some(build_lineup_logprobs(&order, &dists))
    } else {
        None
    };

    Json(ChatResponse {
        choices: vec![Choice {
            message: ResponseMessage {
                role: "assistant".to_string(),
                content,
            },
            logprobs,
            finish_reason: "stop".to_string(),
        }],
        usage: ResponseUsage {
            prompt_tokens: 75,
            completion_tokens: 6 * order.len() as u64,
        },
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Start the fake judge server on a random available port.
///
/// Returns the bound port and a handle to the background task. Abort the handle
/// to shut down the server.
pub async fn start(state: Arc<JudgeState>) -> (u16, tokio::task::JoinHandle<()>) {
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind server");
    let port = listener.local_addr().unwrap().port();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (port, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn one_sample_is_a_hard_judgement() {
        let mut rng = StdRng::seed_from_u64(1);
        let sampled = sample_token_distribution([0.5, 0.5], 1, &mut rng);

        assert_close(sampled.probs.iter().sum(), 1.0);
        assert_close(sampled.probs[sampled.emitted], 1.0);
        assert_close(sampled.probs[1 - sampled.emitted], 0.0);
    }

    #[test]
    fn empirical_probabilities_have_sample_granularity() {
        let samples = 10;
        let mut rng = StdRng::seed_from_u64(2);
        let sampled = sample_token_distribution([0.2, 0.3, 0.5], samples, &mut rng);

        assert_close(sampled.probs.iter().sum(), 1.0);
        assert!(sampled.probs[sampled.emitted] >= 1.0 / samples as f64);
        for probability in sampled.probs {
            let count = probability * samples as f64;
            assert_close(count, count.round());
        }
    }

    #[test]
    fn lineup_samples_second_place_conditionally() {
        let samples = 10;
        let mut rng = StdRng::seed_from_u64(3);
        let (order, dists) = sample_lineup_ranking(&[0.8, 0.1, -0.4], samples, &mut rng);
        let (first, second) = (&dists[0], &dists[1]);

        let mut sorted_order = order.clone();
        sorted_order.sort_unstable();
        assert_eq!(sorted_order, vec![0, 1, 2]);

        assert_close(first.iter().sum(), 1.0);
        assert!(first[order[0]] >= 1.0 / samples as f64);

        // The emitted first-place item is removed before the second sample
        // sample set. The conditional distribution covers exactly the other two.
        assert_close(second[order[0]], 0.0);
        assert_close(second.iter().sum(), 1.0);
        assert!(second[order[1]] >= 1.0 / samples as f64);
    }

    #[test]
    fn lineup_seed_depends_on_every_member() {
        // Two lineups sharing a prefix must not share an RNG stream: if only the
        // last member changes, the sampled ranking has to be independent.
        let base = deterministic_lineup_seed(7, &["a", "b", "c"], 0);
        assert_ne!(base, deterministic_lineup_seed(7, &["a", "b", "d"], 0));
        assert_ne!(base, deterministic_lineup_seed(7, &["a", "c", "b"], 0));
        assert_ne!(base, deterministic_lineup_seed(7, &["a", "b"], 0));
        assert_ne!(base, deterministic_lineup_seed(7, &["a", "b", "c"], 1));
        assert_ne!(base, deterministic_lineup_seed(8, &["a", "b", "c"], 0));
        assert_eq!(base, deterministic_lineup_seed(7, &["a", "b", "c"], 0));
    }

    /// A full ranking at every lineup size: each rank is a valid conditional
    /// distribution over the items still unplaced, and the last is one-hot.
    #[test]
    fn lineup_ranking_is_a_plackett_luce_chain_at_every_size() {
        for size in 2..=9usize {
            let strengths: Vec<f64> = (0..size).map(|i| i as f64 * 0.3 - 1.0).collect();
            let mut rng = StdRng::seed_from_u64(size as u64);
            let (order, dists) = sample_lineup_ranking(&strengths, 20, &mut rng);

            assert_eq!(order.len(), size, "size {size}: short ranking");
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..size).collect::<Vec<_>>(), "size {size}: not a permutation");

            for (rank, dist) in dists.iter().enumerate() {
                assert_close(dist.iter().sum(), 1.0);
                // Items ranked above this one carry no mass.
                for &placed in &order[..rank] {
                    assert_close(dist[placed], 0.0);
                }
                assert!(dist[order[rank]] > 0.0, "size {size}: emitted token has zero mass");
            }
            assert_close(dists[size - 1][order[size - 1]], 1.0);
        }
    }

    #[test]
    fn zero_count_options_are_omitted_from_top_logprobs() {
        let payload = build_logprobs_payload(0, [1.0, 0.0]);
        let top = payload.content[3].top_logprobs.as_ref().unwrap();

        assert_eq!(top.len(), 1);
        assert_eq!(top[0].token, "1");
        assert_close(top[0].logprob, 0.0);
    }

    #[test]
    #[should_panic(expected = "samples_per_judgement must be at least 1")]
    fn zero_samples_panics() {
        let mut rng = StdRng::seed_from_u64(4);
        let _ = sample_token_distribution([0.5, 0.5], 0, &mut rng);
    }
}
