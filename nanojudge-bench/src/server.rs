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

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn handle_chat(
    State(state): State<Arc<JudgeState>>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let prompt = &request.messages.last().expect("empty messages array").content;
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

    let want_logprobs = request.logprobs.unwrap_or(false);

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
