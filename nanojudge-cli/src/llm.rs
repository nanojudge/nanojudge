/// OpenAI-compatible API client for LLM judgements.
use crate::parse::{
    LogprobContent, ParseResult, parse_response, parse_response_text, parse_lineup,
    parse_lineup_text,
};
use crate::prompt::{build_prompt, build_lineup_prompt};
use nanojudge_core::LineupVerdict;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(crate) enum LlmError {
    Retryable(String),
    Permanent(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Retryable(s) | LlmError::Permanent(s) => f.write_str(s),
        }
    }
}

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
    /// Includes reasoning tokens, when the model reasons.
    pub completion_tokens: u64,
    /// Should equal prompt_tokens + completion_tokens. Only used for checking.
    total_tokens: Option<u64>,
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    reasoning_tokens: Option<u64>,
}

impl Usage {
    /// Tokens the model spent reasoning. None if the endpoint didn't report it.
    pub fn reasoning_tokens(&self) -> Option<u64> {
        self.completion_tokens_details.as_ref().and_then(|d| d.reasoning_tokens)
    }

    /// Tokens in the visible answer: completion tokens minus reasoning tokens.
    /// None when reasoning tokens weren't reported.
    pub fn visible_tokens(&self) -> Option<u64> {
        self.reasoning_tokens().map(|r| self.completion_tokens - r)
    }

    /// The `usage` object written to the judgements files.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens(),
            "visible_tokens": self.visible_tokens(),
        })
    }
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
    /// The model's reasoning text. Endpoints use one of these two names; a
    /// missing field and an explicit null both read as None.
    reasoning: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChoiceLogprobs {
    content: Option<Vec<LogprobContent>>,
}

/// Result of a single LLM judgement call.
pub struct PairJudgementResult {
    pub item1_id: i64,
    pub item2_id: i64,
    pub parse_result: ParseResult,
    /// The visible answer. None if the endpoint returned no content.
    pub response_text: Option<String>,
    /// The model's reasoning text. None if the endpoint returned none.
    pub reasoning_text: Option<String>,
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

/// Take the reasoning text from whichever of the two fields the endpoint used.
/// Both being set is an error: there's no way to tell which one is the model's
/// reasoning.
fn pick_reasoning(
    reasoning: Option<String>,
    reasoning_content: Option<String>,
) -> Result<Option<String>, String> {
    match (reasoning, reasoning_content) {
        (Some(_), Some(_)) => Err(
            "the response has both `reasoning` and `reasoning_content`; can't tell which is the model's reasoning".into(),
        ),
        (Some(r), None) | (None, Some(r)) => Ok(Some(r)),
        (None, None) => Ok(None),
    }
}

/// Check the token counts mean what we assume: total tokens are prompt plus
/// completion tokens, and reasoning tokens are part of completion tokens.
/// An endpoint that counts reasoning outside completion tokens fails one of
/// these on a reply that reasons, unless it also leaves reasoning out of the
/// total and reasons less than it answers.
fn check_token_counts(usage: &Usage) -> Result<(), String> {
    if let Some(total) = usage.total_tokens
        && total != usage.prompt_tokens + usage.completion_tokens
    {
        return Err(format!(
            "reported {total} total tokens but {} prompt + {} completion tokens; total tokens should be prompt plus completion tokens",
            usage.prompt_tokens, usage.completion_tokens
        ));
    }
    match usage.reasoning_tokens() {
        Some(r) if r > usage.completion_tokens => Err(format!(
            "reported {r} reasoning tokens but only {} completion tokens; reasoning tokens should be part of completion tokens",
            usage.completion_tokens
        )),
        _ => Ok(()),
    }
}

/// The parts of one LLM reply that the judgement files record.
pub struct LlmReply {
    /// The visible answer. None if the endpoint returned no content.
    pub content: Option<String>,
    /// The model's reasoning text. None if the endpoint returned none.
    pub reasoning: Option<String>,
    pub usage: Option<Usage>,
    /// True if the response was truncated due to hitting max_tokens.
    pub hit_max_tokens: bool,
}

/// Send one chat request to the LLM and return the reply plus the token
/// logprobs (empty in text mode). Returns Err only on HTTP/network failures.
/// Bails (fatal) if logprobs were requested but none came back, if the reply
/// has both `reasoning` and `reasoning_content`, or if its token counts are
/// inconsistent (see `check_token_counts`): these are endpoint problems rather
/// than per-judgement failures.
async fn send_chat_raw(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
) -> Result<(LlmReply, Vec<LogprobContent>), LlmError> {
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

    let resp = req_builder.send().await.map_err(|e| LlmError::Retryable(format!("HTTP request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let msg = format!("LLM API returned {status}: {}", truncate_for_log(&body, 500));
        return Err(if matches!(status.as_u16(), 408 | 409 | 425 | 429) || status.is_server_error() {
            LlmError::Retryable(msg)
        } else {
            LlmError::Permanent(msg)
        });
    }

    let data: ChatCompletionResponse = resp
        .json()
        .await
        .map_err(|e| LlmError::Retryable(format!("Failed to parse LLM response JSON: {e}")))?;

    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or(LlmError::Retryable("No choices in LLM response".into()))?;

    let message = choice.message;
    let reasoning = pick_reasoning(message.reasoning, message.reasoning_content)
        .unwrap_or_else(|e| crate::bail(format!("{}: {e}", config.model)));
    if let Some(ref usage) = data.usage {
        check_token_counts(usage).unwrap_or_else(|e| crate::bail(format!("{}: {e}", config.model)));
    }
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

    let reply = LlmReply {
        content: message.content,
        reasoning,
        usage: data.usage,
        hit_max_tokens,
    };
    Ok((reply, logprobs))
}

/// Send one HTTP request to the LLM and parse the pairwise verdict.
/// Returns Ok on any successful HTTP response (even if verdict is unparseable).
/// Returns Err only on HTTP/network failures.
pub async fn send_pair_judgement_request(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
    min_logprob_coverage: f64,
) -> Result<(ParseResult, LlmReply), LlmError> {
    let (reply, logprobs) = send_chat_raw(client, config, prompt).await?;

    let parse_result = if config.logprobs {
        parse_response(&logprobs, min_logprob_coverage)
    } else {
        match &reply.content {
            Some(content) => parse_response_text(content),
            None => ParseResult { category_probs: None },
        }
    };

    Ok((parse_result, reply))
}

/// Call the LLM to compare two items, with retries on HTTP errors.
///
/// Retries up to `max_retries` times with exponential backoff (1s, 4s, 16s).
/// Only HTTP/network errors trigger retries — unparseable verdicts do not.
#[allow(clippy::too_many_arguments)]
pub async fn judge_pair(
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
    deliberation_length: &str,
    max_retries: usize,
    verbose: bool,
    judge_name: &str,
) -> Result<PairJudgementResult, String> {
    let prompt = build_prompt(template, criterion, item1_name, item2_name, item1_title, item2_title, deliberation_length);

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        match send_pair_judgement_request(client, config, &prompt, min_logprob_coverage).await {
            Ok((parse_result, reply)) => {
                return Ok(PairJudgementResult {
                    item1_id,
                    item2_id,
                    parse_result,
                    response_text: reply.content,
                    reasoning_text: reply.reasoning,
                    prompt: prompt.clone(),
                    retries_used: attempt,
                    usage: reply.usage,
                    hit_max_tokens: reply.hit_max_tokens,
                });
            }
            Err(LlmError::Permanent(e)) => {
                return Err(e);
            }
            Err(LlmError::Retryable(e)) => {
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

/// Result of a single lineup judgement call.
pub struct LineupJudgementResult {
    /// The lineup's item IDs, in presentation order (slot A first).
    pub item_ids: Vec<i64>,
    /// The judge's ranking and place probabilities, indexing into `item_ids`.
    /// None if the response was unparseable.
    pub verdict: Option<LineupVerdict>,
    /// The visible answer. None if the endpoint returned no content.
    pub response_text: Option<String>,
    /// The model's reasoning text. None if the endpoint returned none.
    pub reasoning_text: Option<String>,
    pub prompt: String,
    pub retries_used: usize,
    pub usage: Option<Usage>,
    pub hit_max_tokens: bool,
}

/// Send one HTTP request for a lineup judgement and parse the response into a
/// verdict over the lineup's options. Returns Err only on HTTP/network
/// failures; an unparseable ranking yields `Ok` with `verdict = None`.
async fn send_lineup_judgement_request(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
    lineup_size: usize,
    min_logprob_coverage: f64,
) -> Result<(Option<LineupVerdict>, LlmReply), LlmError> {
    let (reply, logprobs) = send_chat_raw(client, config, prompt).await?;

    let verdict = if config.logprobs {
        parse_lineup(&logprobs, lineup_size, min_logprob_coverage)
    } else {
        reply.content.as_deref().and_then(|content| parse_lineup_text(content, lineup_size))
    };

    Ok((verdict, reply))
}

/// Call the LLM to rank a lineup's items, with retries on HTTP errors. Mirrors
/// `judge_pair`: retries only HTTP/network errors, never an unparseable
/// ranking. `option_texts` and `item_ids` are in presentation order and must be
/// the same length.
#[allow(clippy::too_many_arguments)]
pub async fn judge_lineup(
    client: &Client,
    config: &LlmConfig,
    template: &str,
    criterion: &str,
    option_texts: &[&str],
    item_ids: &[i64],
    min_logprob_coverage: f64,
    deliberation_length: &str,
    max_retries: usize,
    verbose: bool,
    judge_name: &str,
) -> Result<LineupJudgementResult, String> {
    assert_eq!(
        option_texts.len(),
        item_ids.len(),
        "option_texts and item_ids must describe the same lineup"
    );
    let lineup_size = item_ids.len();
    let prompt = build_lineup_prompt(template, criterion, option_texts, deliberation_length);

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        match send_lineup_judgement_request(client, config, &prompt, lineup_size, min_logprob_coverage).await {
            Ok((verdict, reply)) => {
                return Ok(LineupJudgementResult {
                    item_ids: item_ids.to_vec(),
                    verdict,
                    response_text: reply.content,
                    reasoning_text: reply.reasoning,
                    prompt: prompt.clone(),
                    retries_used: attempt,
                    usage: reply.usage,
                    hit_max_tokens: reply.hit_max_tokens,
                });
            }
            Err(LlmError::Permanent(e)) => {
                return Err(e);
            }
            Err(LlmError::Retryable(e)) => {
                last_err = e;
                if attempt < max_retries {
                    if verbose {
                        eprintln!(
                            "  Retry {}/{} for lineup [{}]: {}",
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

    fn usage(json: &str) -> Usage {
        serde_json::from_str(json).unwrap()
    }

    fn message(json: &str) -> MessageContent {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_usage_reasoning_tokens_reported() {
        let u = usage(r#"{"prompt_tokens": 13, "completion_tokens": 26,
            "completion_tokens_details": {"reasoning_tokens": 19}}"#);
        assert_eq!(u.reasoning_tokens(), Some(19));
        assert_eq!(u.visible_tokens(), Some(7));
    }

    #[test]
    fn test_usage_zero_reasoning_tokens_is_not_null() {
        let u = usage(r#"{"prompt_tokens": 13, "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 0}}"#);
        let j = u.to_json();
        assert_eq!(j["reasoning_tokens"], serde_json::json!(0));
        assert_eq!(j["visible_tokens"], serde_json::json!(12));
    }

    #[test]
    fn test_usage_reasoning_tokens_not_reported() {
        for json in [
            r#"{"prompt_tokens": 5, "completion_tokens": 9}"#,
            r#"{"prompt_tokens": 5, "completion_tokens": 9, "completion_tokens_details": null}"#,
            r#"{"prompt_tokens": 5, "completion_tokens": 9, "completion_tokens_details": {}}"#,
            r#"{"prompt_tokens": 5, "completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": null}}"#,
        ] {
            let j = usage(json).to_json();
            assert!(j["reasoning_tokens"].is_null(), "{json}");
            assert!(j["visible_tokens"].is_null(), "{json}");
            assert_eq!(j["completion_tokens"], serde_json::json!(9));
        }
    }

    #[test]
    fn test_check_token_counts_reasoning() {
        let within = usage(r#"{"prompt_tokens": 1, "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 10}}"#);
        assert!(check_token_counts(&within).is_ok());
        let over = usage(r#"{"prompt_tokens": 1, "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 11}}"#);
        assert!(check_token_counts(&over).is_err());
        let unreported = usage(r#"{"prompt_tokens": 1, "completion_tokens": 10}"#);
        assert!(check_token_counts(&unreported).is_ok());
    }

    #[test]
    fn test_check_token_counts_total() {
        // Gemini's reply from OpenRouter: reasoning inside completion.
        let matching = usage(r#"{"prompt_tokens": 10, "completion_tokens": 174, "total_tokens": 184,
            "completion_tokens_details": {"reasoning_tokens": 162}}"#);
        assert!(check_token_counts(&matching).is_ok());
        // Reasoning counted outside completion but inside the total.
        let separate = usage(r#"{"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 184,
            "completion_tokens_details": {"reasoning_tokens": 162}}"#);
        assert!(check_token_counts(&separate).is_err());
        let off_by_one = usage(r#"{"prompt_tokens": 10, "completion_tokens": 174, "total_tokens": 185}"#);
        assert!(check_token_counts(&off_by_one).is_err());
        let unreported = usage(r#"{"prompt_tokens": 10, "completion_tokens": 174}"#);
        assert!(check_token_counts(&unreported).is_ok());
        let null = usage(r#"{"prompt_tokens": 10, "completion_tokens": 174, "total_tokens": null}"#);
        assert!(check_token_counts(&null).is_ok());
    }

    #[test]
    fn test_message_missing_and_null_fields_are_none() {
        let missing = message(r#"{"role": "assistant"}"#);
        assert_eq!(missing.content, None);
        assert_eq!(missing.reasoning, None);
        assert_eq!(missing.reasoning_content, None);
        let null = message(r#"{"content": null, "reasoning": null, "reasoning_content": null}"#);
        assert_eq!(null.content, None);
        assert_eq!(null.reasoning, None);
        assert_eq!(null.reasoning_content, None);
    }

    #[test]
    fn test_message_empty_strings_stay_empty() {
        let m = message(r#"{"content": "", "reasoning": ""}"#);
        assert_eq!(m.content.as_deref(), Some(""));
        assert_eq!(pick_reasoning(m.reasoning, m.reasoning_content), Ok(Some(String::new())));
    }

    #[test]
    fn test_pick_reasoning() {
        let s = |t: &str| Some(t.to_string());
        assert_eq!(pick_reasoning(s("a"), None), Ok(s("a")));
        assert_eq!(pick_reasoning(None, s("b")), Ok(s("b")));
        assert_eq!(pick_reasoning(None, None), Ok(None));
        assert!(pick_reasoning(s("a"), s("b")).is_err());
        // An empty string still counts as present.
        assert!(pick_reasoning(s(""), s("b")).is_err());
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
