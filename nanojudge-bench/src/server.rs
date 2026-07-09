/// Fake OpenAI-compatible endpoint backed by a Bradley-Terry strength table.
///
/// Receives chat completion requests, extracts the two item IDs from the prompt,
/// looks up their true strengths, flips a BT-weighted coin, and returns a verdict
/// in either text-only or logprobs format — matching the real OpenAI response shape
/// that the NanoJudge CLI expects.
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
    /// Per-pair encounter counter so repeated matchups get independent coin flips.
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

/// Bradley-Terry coin flip: returns 'A' (item1 wins) or 'D' (item2 wins).
fn verdict_letter(s1: f64, s2: f64, rng: &mut impl Rng) -> char {
    let p = 1.0 / (1.0 + (-(s1 - s2)).exp());
    if rng.random::<f64>() < p {
        'A'
    } else {
        'D'
    }
}

/// Derive a deterministic seed from (base_seed, item1, item2, encounter).
///
/// Items are hashed in prompt order so positional swaps produce different seeds.
/// The encounter counter ensures repeated matchups of the same ordered pair get
/// independent coin flips.
fn deterministic_pair_seed(base: u64, item1: &str, item2: &str, seq: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    base.hash(&mut hasher);
    item1.hash(&mut hasher);
    item2.hash(&mut hasher);
    seq.hash(&mut hasher);
    hasher.finish()
}

fn verdict_text(letter: char) -> &'static str {
    match letter {
        'A' => "Verdict A: Option 1, clearly",
        'D' => "Verdict D: Option 2, clearly",
        _ => unreachable!(),
    }
}

/// Build a one-hot logprobs payload for the given verdict letter.
///
/// The winning letter gets logprob 0.0 (probability 1.0), all others get -100.0
/// (probability ~0). This is a valid logprobs response — real LLMs produce similar
/// one-hot distributions when they are highly confident.
fn build_logprobs_payload(letter: char) -> ChoiceLogprobs {
    let verdict_suffix = match letter {
        'A' => ": Option 1, clearly",
        'D' => ": Option 2, clearly",
        _ => unreachable!(),
    };

    let top_logprobs: Vec<TopLogprobEntry> = ['A', 'B', 'C', 'D']
        .iter()
        .map(|&l| TopLogprobEntry {
            token: l.to_string(),
            logprob: if l == letter { 0.0 } else { -100.0 },
        })
        .collect();

    ChoiceLogprobs {
        content: vec![
            LogprobToken {
                token: "Verdict".to_string(),
                top_logprobs: None,
            },
            LogprobToken {
                token: format!(" {letter}"),
                top_logprobs: Some(top_logprobs),
            },
            LogprobToken {
                token: verdict_suffix.to_string(),
                top_logprobs: None,
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

/// Sample an index from a length-3 probability vector.
fn sample3(p: [f64; 3], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.random();
    if r < p[0] {
        0
    } else if r < p[0] + p[1] {
        1
    } else {
        2
    }
}

/// A three-way ranking drawn from the Plackett-Luce model on the three
/// strengths: sample 1st place ∝ softmax(strength), then 2nd place ∝ softmax
/// among the remaining two. Returns the sampled order (indices into A,B,C), the
/// 1st-place distribution `p1` over all three, and the 2nd-place distribution
/// `p2` over the three (mass only on the two non-winners).
fn three_way_ranking(
    strengths: [f64; 3],
    rng: &mut impl Rng,
) -> ([usize; 3], [f64; 3], [f64; 3]) {
    let p1 = softmax3(strengths);
    let first = sample3(p1, rng);

    // Remaining two, softmax over just their strengths.
    let rest: Vec<usize> = (0..3).filter(|&i| i != first).collect();
    let (i, j) = (rest[0], rest[1]);
    let m = strengths[i].max(strengths[j]);
    let (ei, ej) = ((strengths[i] - m).exp(), (strengths[j] - m).exp());
    let (pi, pj) = (ei / (ei + ej), ej / (ei + ej));

    let second_is_i = rng.random::<f64>() < pi;
    let second = if second_is_i { i } else { j };
    let third = if second_is_i { j } else { i };

    let mut p2 = [0.0_f64; 3];
    p2[i] = pi;
    p2[j] = pj;

    ([first, second, third], p1, p2)
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
/// Option" line ends in a letter token carrying that slot's distribution over
/// A/B/C: 1st gets `p1`, 2nd gets `p2` (mass on the two non-winners), 3rd is
/// one-hot on the remaining option. "Option" is the parser's anchor.
fn build_three_way_logprobs(order: [usize; 3], p1: [f64; 3], p2: [f64; 3]) -> ChoiceLogprobs {
    let mut p3 = [0.0_f64; 3];
    p3[order[2]] = 1.0;

    let slot = |ordinal: &str, letter_idx: usize, dist: [f64; 3]| -> Vec<LogprobToken> {
        let top: Vec<TopLogprobEntry> = (0..3)
            .map(|i| TopLogprobEntry {
                token: RANK_LETTERS[i].to_string(),
                logprob: if dist[i] > 0.0 { dist[i].ln() } else { -100.0 },
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

    let letter = {
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
        verdict_letter(s1, s2, &mut rng)
    };

    Json(ChatResponse {
        choices: vec![Choice {
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: verdict_text(letter).to_string(),
            },
            logprobs: if want_logprobs {
                Some(build_logprobs_payload(letter))
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

/// Handle a three-way comparison request: draw a Plackett-Luce ranking from the
/// three items' strengths and return it (with per-slot logprobs when requested).
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

    let (order, p1, p2) = three_way_ranking([sa, sb, sc], &mut rng);
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
