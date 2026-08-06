/// Fake OpenAI-compatible endpoint backed by a Bradley-Terry strength table.
///
/// Receives chat completion requests, extracts the item IDs from the prompt,
/// looks up their true strengths, and estimates each verdict-token distribution
/// with repeated Bradley-Terry/Plackett-Luce samples. The empirical distribution
/// is returned in the OpenAI logprobs response shape that the real NanoJudge CLI
/// expects, while the whole request still counts as one comparison.
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
    pub samples_per_comparison: usize,
    /// Per-pair encounter counter so repeated matchups get independent sample batches.
    pub pair_counts: Mutex<HashMap<(String, String), u64>>,
}

// ---------------------------------------------------------------------------
// Prompt parsing
// ---------------------------------------------------------------------------

/// Extract the two item texts from a NanoJudge comparison prompt.
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

/// True if this is a three-way (3-item) comparison prompt.
fn is_three_way(prompt: &str) -> bool {
    prompt.contains("Option A:\n")
}

/// Extract the three item texts from a three-way prompt. Mirrors `extract_items`
/// but for the "Option A:/Option B:/Option C:" layout.
fn extract_three_items(prompt: &str) -> (String, String, String) {
    let a_marker = "Option A:\n";
    let b_marker = "Option B:\n";
    let c_marker = "Option C:\n";

    let a_start = prompt.find(a_marker).expect("prompt missing 'Option A:' marker") + a_marker.len();
    let b_pos = prompt.find(b_marker).expect("prompt missing 'Option B:' marker");
    let item_a = prompt[a_start..b_pos].trim().to_string();

    let b_start = b_pos + b_marker.len();
    let c_pos = prompt.find(c_marker).expect("prompt missing 'Option C:' marker");
    let item_b = prompt[b_start..c_pos].trim().to_string();

    let c_start = c_pos + c_marker.len();
    let c_end = prompt[c_start..]
        .find("\n\nInstructions:")
        .map(|p| c_start + p)
        .unwrap_or(prompt.len());
    let item_c = prompt[c_start..c_end].trim().to_string();

    (item_a, item_b, item_c)
}

// ---------------------------------------------------------------------------
// Verdict generation
// ---------------------------------------------------------------------------

/// One emitted verdict token plus the empirical distribution estimated from a
/// batch of independent draws. The emitted token is the first draw in the
/// batch, matching normal generation while guaranteeing it has nonzero mass in
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
    assert!(samples > 0, "samples_per_comparison must be at least 1");

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

/// Derive a deterministic seed from (base_seed, item1, item2, encounter).
///
/// Items are hashed in prompt order so positional swaps produce different seeds.
/// The encounter counter ensures repeated matchups of the same ordered pair get
/// independent sample batches.
fn deterministic_pair_seed(base: u64, item1: &str, item2: &str, seq: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    base.hash(&mut hasher);
    item1.hash(&mut hasher);
    item2.hash(&mut hasher);
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

/// Softmax over three strengths, numerically stabilized.
fn softmax3(vals: [f64; 3]) -> [f64; 3] {
    let m = vals[0].max(vals[1]).max(vals[2]);
    let e = [(vals[0] - m).exp(), (vals[1] - m).exp(), (vals[2] - m).exp()];
    let sum = e[0] + e[1] + e[2];
    [e[0] / sum, e[1] / sum, e[2] / sum]
}

/// Estimate the two autoregressive token distributions of a three-way ranking.
/// First place gets `samples` draws from all three Plackett-Luce weights. The
/// first draw is the emitted winner. Conditional on that emitted token, second
/// place gets a fresh `samples` draws from the two remaining items; its first
/// draw is emitted second. Third place is the one remaining item.
fn sample_three_way_ranking(
    strengths: [f64; 3],
    samples: usize,
    rng: &mut impl Rng,
) -> ([usize; 3], [f64; 3], [f64; 3]) {
    let first = sample_token_distribution(softmax3(strengths), samples, rng);

    // Remaining two, softmax over just their strengths, then independently
    // sample the conditional second-place token distribution.
    let rest: Vec<usize> = (0..3).filter(|&i| i != first.emitted).collect();
    let (i, j) = (rest[0], rest[1]);
    let m = strengths[i].max(strengths[j]);
    let (ei, ej) = ((strengths[i] - m).exp(), (strengths[j] - m).exp());
    let (pi, pj) = (ei / (ei + ej), ej / (ei + ej));
    let second_local = sample_token_distribution([pi, pj], samples, rng);
    let second = if second_local.emitted == 0 { i } else { j };
    let third = if second_local.emitted == 0 { j } else { i };

    let mut p2 = [0.0_f64; 3];
    p2[i] = second_local.probs[0];
    p2[j] = second_local.probs[1];

    ([first.emitted, second, third], first.probs, p2)
}

const RANK_LETTERS: [char; 3] = ['A', 'B', 'C'];

/// Render the three-way ranking as the "Nth place is Option X" lines the
/// default three-way template asks for.
fn three_way_text(order: [usize; 3]) -> String {
    format!(
        "First place is Option {}\nSecond place is Option {}\nThird place is Option {}",
        RANK_LETTERS[order[0]], RANK_LETTERS[order[1]], RANK_LETTERS[order[2]],
    )
}

/// Build the logprobs payload for a three-way ranking. Each "Nth place is
/// Option" line ends in a letter token carrying that slot's empirical
/// distribution over A/B/C: 1st gets `p1`, 2nd gets the conditional `p2`, and
/// 3rd is one-hot on the remaining option. "Option" is the parser's anchor.
fn build_three_way_logprobs(order: [usize; 3], p1: [f64; 3], p2: [f64; 3]) -> ChoiceLogprobs {
    let mut p3 = [0.0_f64; 3];
    p3[order[2]] = 1.0;

    let slot = |ordinal: &str, letter_idx: usize, dist: [f64; 3]| -> Vec<LogprobToken> {
        let top: Vec<TopLogprobEntry> = (0..3)
            .filter(|&i| dist[i] > 0.0)
            .map(|i| TopLogprobEntry {
                token: RANK_LETTERS[i].to_string(),
                logprob: dist[i].ln(),
            })
            .collect();
        vec![
            LogprobToken { token: ordinal.to_string(), top_logprobs: None },
            LogprobToken { token: " place".to_string(), top_logprobs: None },
            LogprobToken { token: " is".to_string(), top_logprobs: None },
            LogprobToken { token: " Option".to_string(), top_logprobs: None },
            LogprobToken { token: format!(" {}", RANK_LETTERS[letter_idx]), top_logprobs: Some(top) },
            LogprobToken { token: "\n".to_string(), top_logprobs: None },
        ]
    };

    let mut content = Vec::new();
    content.extend(slot("First", order[0], p1));
    content.extend(slot("Second", order[1], p2));
    content.extend(slot("Third", order[2], p3));
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

    if is_three_way(prompt) {
        return handle_three_way(state, prompt, want_logprobs);
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
            let mut counts = state.pair_counts.lock().unwrap();
            let n = counts.entry(key).or_insert(0);
            let current = *n;
            *n += 1;
            current
        };
        let pair_seed = deterministic_pair_seed(state.seed, &item1, &item2, encounter);
        let mut rng = StdRng::seed_from_u64(pair_seed);
        let p1 = 1.0 / (1.0 + (-(s1 - s2)).exp());
        sample_token_distribution(
            [p1, 1.0 - p1],
            state.samples_per_comparison,
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

/// Handle a three-way comparison request: estimate the first-place distribution
/// from repeated three-way draws, declare its first draw as the emitted winner,
/// then independently estimate second place from repeated draws conditional on
/// that emitted winner. Third place is deterministic.
fn handle_three_way(
    state: Arc<JudgeState>,
    prompt: &str,
    want_logprobs: bool,
) -> Json<ChatResponse> {
    let (item_a, item_b, item_c) = extract_three_items(prompt);
    let sa = *state.strengths.get(&item_a).unwrap_or_else(|| panic!("unknown item: {item_a:?}"));
    let sb = *state.strengths.get(&item_b).unwrap_or_else(|| panic!("unknown item: {item_b:?}"));
    let sc = *state.strengths.get(&item_c).unwrap_or_else(|| panic!("unknown item: {item_c:?}"));

    // Independent draw per repeated (ordered) triple, keyed via the shared
    // encounter counter (the second key slot is unused for triples).
    let key = (format!("{item_a}\u{0}{item_b}\u{0}{item_c}"), String::new());
    let encounter = {
        let mut counts = state.pair_counts.lock().unwrap();
        let n = counts.entry(key).or_insert(0);
        let current = *n;
        *n += 1;
        current
    };
    let seed = deterministic_pair_seed(state.seed, &item_a, &item_b, encounter.wrapping_add(sc.to_bits()));
    let mut rng = StdRng::seed_from_u64(seed);

    let (order, p1, p2) = sample_three_way_ranking(
        [sa, sb, sc],
        state.samples_per_comparison,
        &mut rng,
    );
    let content = three_way_text(order);
    let logprobs = if want_logprobs {
        Some(build_three_way_logprobs(order, p1, p2))
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
            completion_tokens: 18,
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
    fn one_sample_is_a_hard_judgment() {
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
    fn three_way_samples_second_place_conditionally() {
        let samples = 10;
        let mut rng = StdRng::seed_from_u64(3);
        let (order, first, second) =
            sample_three_way_ranking([0.8, 0.1, -0.4], samples, &mut rng);

        let mut sorted_order = order;
        sorted_order.sort_unstable();
        assert_eq!(sorted_order, [0, 1, 2]);

        assert_close(first.iter().sum(), 1.0);
        assert!(first[order[0]] >= 1.0 / samples as f64);

        // The emitted first-place item is removed before the second sample
        // batch. The conditional distribution covers exactly the other two.
        assert_close(second[order[0]], 0.0);
        assert_close(second.iter().sum(), 1.0);
        assert!(second[order[1]] >= 1.0 / samples as f64);
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
    #[should_panic(expected = "samples_per_comparison must be at least 1")]
    fn zero_samples_panics() {
        let mut rng = StdRng::seed_from_u64(4);
        let _ = sample_token_distribution([0.5, 0.5], 0, &mut rng);
    }
}
