/// OpenAI-compatible API client for pairwise comparisons.
use crate::parse::{
    LogprobContent, ParseResult, parse_response, parse_response_text, parse_three_way,
    parse_three_way_text,
};
use crate::prompt::{build_prompt, build_three_way_prompt};
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let truncated = &s[..s.floor_char_boundary(max)];
    format!("{}...[{} chars truncated]", truncated, s.len() - truncated.len())
}

/// Configuration for the LLM endpoint.
pub struct LlmConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: f64,
    /// Standard deviation of temperature jitter. 0.0 = no jitter (default).
    /// Uses N(1.0, jitter) multiplier clamped to [0.8, 1.2].
    pub temperature_jitter: f64,
    /// Presence penalty: penalizes repeated tokens. Only sent if Some.
    pub presence_penalty: Option<f64>,
    /// Top-p (nucleus sampling). Only sent if Some.
    pub top_p: Option<f64>,
    /// When true, extract logprobs for continuous win probabilities.
    pub logprobs: bool,
    /// Maximum tokens in the LLM response.
    pub max_tokens: u32,
    /// OpenRouter extension: reasoning effort level (e.g. "none" to disable Qwen thinking).
    pub reasoning_effort: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ReasoningConfig {
    effort: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<&'static str>,
    /// vLLM extension: include the stop string in the output text.
    #[serde(skip_serializing_if = "Option::is_none")]
    include_stop_str_in_output: Option<bool>,
    /// OpenRouter extension: controls reasoning/thinking mode.
    /// Used to disable chain-of-thought for models like Qwen.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageContent,
    logprobs: Option<ChoiceLogprobs>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChoiceLogprobs {
    content: Option<Vec<LogprobContent>>,
}

/// Result of a single LLM comparison call.
pub struct ComparisonResult {
    pub item1_id: i64,
    pub item2_id: i64,
    pub parse_result: ParseResult,
    pub response_text: String,
    pub prompt: String,
    pub retries_used: usize,
    pub usage: Option<Usage>,
    /// True if the response was truncated due to hitting max_tokens.
    pub hit_max_tokens: bool,
}

/// Build the full chat completions URL from a user-provided endpoint.
///
/// If the endpoint has no path (just `scheme://host` or `scheme://host:port`),
/// assumes OpenAI-style and appends `/v1/chat/completions`. Otherwise appends
/// `/chat/completions` to whatever path the user provided.
fn build_completions_url(endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    // Find the start of the path: skip past "scheme://host(:port)"
    let after_scheme = base.find("://").map(|i| i + 3).unwrap_or(0);
    let has_path = base[after_scheme..].contains('/');
    if has_path {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

/// Apply normal jitter to temperature: N(1.0, jitter_std) clamped to [0.8, 1.2].
/// Uses Box-Muller transform to avoid an extra crate dependency.
/// Returns base unchanged if jitter_std is 0.0.
pub(crate) fn jittered_temperature(base: f64, jitter_std: f64, rng: &mut impl Rng) -> f64 {
    if jitter_std == 0.0 {
        return base;
    }
    let u1: f64 = rng.random::<f64>().max(1e-10);
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let multiplier = (1.0 + jitter_std * z).clamp(0.8, 1.2);
    base * multiplier
}

/// Send one chat request to the LLM and return the raw pieces the callers need:
/// the response text, the token logprobs (empty in text mode), token usage, and
/// whether the response hit `max_tokens`. Returns Err only on HTTP/network
/// failures. Bails (fatal) if logprobs were requested but none came back, since
/// that is a misconfiguration rather than a per-comparison failure.
async fn send_chat_raw(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
) -> Result<(String, Vec<LogprobContent>, Option<Usage>, bool), String> {
    let request = ChatCompletionRequest {
        model: config.model.clone(),
        messages: vec![ChatMessage {
            role: "user",
            content: prompt.to_string(),
        }],
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        logprobs: if config.logprobs { Some(true) } else { None },
        top_logprobs: if config.logprobs { Some(10) } else { None },
        presence_penalty: config.presence_penalty,
        top_p: config.top_p,
        stop: vec![],
        include_stop_str_in_output: None,
        reasoning: config.reasoning_effort.as_ref().map(|effort| ReasoningConfig {
            effort: effort.clone(),
        }),
        chat_template_kwargs: config.chat_template_kwargs.clone(),
    };

    let url = build_completions_url(&config.endpoint);

    let mut req_builder = client.post(&url).json(&request);
    if let Some(ref key) = config.api_key {
        req_builder = req_builder.bearer_auth(key);
    }

    let resp = req_builder.send().await.map_err(|e| format!("HTTP request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("LLM API returned {status}: {}", truncate_for_log(&body, 500)));
    }

    let data: ChatCompletionResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse LLM response JSON: {e}"))?;

    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or("No choices in LLM response")?;

    let content = choice.message.content.unwrap_or_default();
    let hit_max_tokens = choice.finish_reason.as_deref() == Some("length");

    let logprobs = if config.logprobs {
        let lp = choice.logprobs.and_then(|lp| lp.content).unwrap_or_default();
        if lp.is_empty() {
            crate::bail(format!("{} returned no logprobs. If your endpoint does not support logprobs, disable logprobs in your config.", config.model));
        }
        lp
    } else {
        Vec::new()
    };

    Ok((content, logprobs, data.usage, hit_max_tokens))
}

/// Send one HTTP request to the LLM and parse the pairwise verdict.
/// Returns Ok on any successful HTTP response (even if verdict is unparseable).
/// Returns Err only on HTTP/network failures.
pub async fn send_comparison_request(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
    min_logprob_coverage: f64,
) -> Result<(ParseResult, String, Option<Usage>, bool), String> {
    let (content, logprobs, usage, hit_max_tokens) = send_chat_raw(client, config, prompt).await?;

    let parse_result = if config.logprobs {
        parse_response(&logprobs, min_logprob_coverage)
    } else {
        parse_response_text(&content)
    };

    Ok((parse_result, content, usage, hit_max_tokens))
}

/// Call the LLM to compare two items, with retries on HTTP errors.
///
/// Retries up to `max_retries` times with exponential backoff (1s, 4s, 16s).
/// Only HTTP/network errors trigger retries — unparseable verdicts do not.
#[allow(clippy::too_many_arguments)]
pub async fn compare_pair(
    client: &Client,
    config: &LlmConfig,
    template: &str,
    criterion: &str,
    item1_name: &str,
    item2_name: &str,
    item1_title: &str,
    item2_title: &str,
    item1_id: i64,
    item2_id: i64,
    min_logprob_coverage: f64,
    analysis_length: &str,
    max_retries: usize,
    verbose: bool,
    judge_name: &str,
) -> Result<ComparisonResult, String> {
    let prompt = build_prompt(template, criterion, item1_name, item2_name, item1_title, item2_title, analysis_length);

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        match send_comparison_request(client, config, &prompt, min_logprob_coverage).await {
            Ok((parse_result, content, usage, hit_max_tokens)) => {
                return Ok(ComparisonResult {
                    item1_id,
                    item2_id,
                    parse_result,
                    response_text: content,
                    prompt: prompt.clone(),
                    retries_used: attempt,
                    usage,
                    hit_max_tokens,
                });
            }
            Err(e) => {
                last_err = e;
                if attempt < max_retries {
                    if verbose {
                        eprintln!(
                            "  Retry {}/{} for {} vs {} [{}]: {}",
                            attempt + 1, max_retries,
                            truncate_for_log(item1_name, 200),
                            truncate_for_log(item2_name, 200),
                            judge_name, last_err
                        );
                    }
                    let backoff = std::time::Duration::from_secs(4u64.pow(attempt as u32).min(16));
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }

    Err(last_err)
}

/// Result of a single 3-way (three-item) comparison call.
pub struct ThreeWayResult {
    pub item_a_id: i64,
    pub item_b_id: i64,
    pub item_c_id: i64,
    /// Winner-distribution `[q_A, q_B, q_C]` (probability each option is best),
    /// aligned to items a, b, c. None if the response was unparseable.
    pub winner_dist: Option<[f64; 3]>,
    pub response_text: String,
    pub prompt: String,
    pub retries_used: usize,
    pub usage: Option<Usage>,
    pub hit_max_tokens: bool,
}

/// Send one HTTP request for a 3-way comparison and fold the response into a
/// winner-distribution over the three options. Returns Err only on HTTP/network
/// failures; an unparseable ranking yields `Ok` with `winner_dist = None`.
async fn send_three_way_request(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
    min_logprob_coverage: f64,
) -> Result<(Option<[f64; 3]>, String, Option<Usage>, bool), String> {
    let (content, logprobs, usage, hit_max_tokens) = send_chat_raw(client, config, prompt).await?;

    let winner_dist = if config.logprobs {
        parse_three_way(&logprobs, min_logprob_coverage)
    } else {
        parse_three_way_text(&content)
    };

    Ok((winner_dist, content, usage, hit_max_tokens))
}

/// Call the LLM to rank three items, with retries on HTTP errors. Mirrors
/// `compare_pair`: retries only HTTP/network errors, never an unparseable
/// ranking.
#[allow(clippy::too_many_arguments)]
pub async fn compare_triple(
    client: &Client,
    config: &LlmConfig,
    template: &str,
    criterion: &str,
    option_a: &str,
    option_b: &str,
    option_c: &str,
    item_a_id: i64,
    item_b_id: i64,
    item_c_id: i64,
    min_logprob_coverage: f64,
    analysis_length: &str,
    max_retries: usize,
    verbose: bool,
    judge_name: &str,
) -> Result<ThreeWayResult, String> {
    let prompt = build_three_way_prompt(template, criterion, option_a, option_b, option_c, analysis_length);

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        match send_three_way_request(client, config, &prompt, min_logprob_coverage).await {
            Ok((winner_dist, content, usage, hit_max_tokens)) => {
                return Ok(ThreeWayResult {
                    item_a_id,
                    item_b_id,
                    item_c_id,
                    winner_dist,
                    response_text: content,
                    prompt: prompt.clone(),
                    retries_used: attempt,
                    usage,
                    hit_max_tokens,
                });
            }
            Err(e) => {
                last_err = e;
                if attempt < max_retries {
                    if verbose {
                        eprintln!(
                            "  Retry {}/{} for 3-way [{}]: {}",
                            attempt + 1, max_retries, judge_name, last_err
                        );
                    }
                    let backoff = std::time::Duration::from_secs(4u64.pow(attempt as u32).min(16));
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }

    Err(last_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jittered_temperature_no_jitter() {
        let mut rng = rand::rng();
        assert_eq!(jittered_temperature(0.7, 0.0, &mut rng), 0.7);
        assert_eq!(jittered_temperature(1.0, 0.0, &mut rng), 1.0);
        assert_eq!(jittered_temperature(0.0, 0.0, &mut rng), 0.0);
    }

    #[test]
    fn test_jittered_temperature_stays_in_range() {
        let mut rng = rand::rng();
        let base = 0.7;
        for _ in 0..1000 {
            let result = jittered_temperature(base, 0.1, &mut rng);
            assert!(result >= base * 0.8, "result {result} < {}", base * 0.8);
            assert!(result <= base * 1.2, "result {result} > {}", base * 1.2);
        }
    }

    #[test]
    fn test_jittered_temperature_high_jitter_still_clamped() {
        let mut rng = rand::rng();
        let base = 1.0;
        for _ in 0..1000 {
            let result = jittered_temperature(base, 10.0, &mut rng);
            assert!(result >= base * 0.8);
            assert!(result <= base * 1.2);
        }
    }

    #[test]
    fn test_build_url_bare_host() {
        assert_eq!(
            build_completions_url("http://localhost:8000"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_url_bare_host_trailing_slash() {
        assert_eq!(
            build_completions_url("http://localhost:8000/"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_url_with_v1() {
        assert_eq!(
            build_completions_url("http://localhost:8000/v1"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_url_with_v1_trailing_slash() {
        assert_eq!(
            build_completions_url("http://localhost:8000/v1/"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_url_openai() {
        assert_eq!(
            build_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_url_openrouter() {
        assert_eq!(
            build_completions_url("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_url_deepinfra() {
        // The bug: path after /v1 used to get /v1 appended again
        assert_eq!(
            build_completions_url("https://api.deepinfra.com/v1/openai"),
            "https://api.deepinfra.com/v1/openai/chat/completions"
        );
    }
}
