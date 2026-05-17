/// OpenAI-compatible API client for pairwise comparisons.
use crate::parse::{LogprobContent, ParseResult, parse_response, parse_response_text};
use crate::prompt::build_prompt;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};

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
fn jittered_temperature(base: f64, jitter_std: f64) -> f64 {
    if jitter_std == 0.0 {
        return base;
    }
    let mut rng = rand::rng();
    let u1: f64 = rng.random::<f64>().max(1e-10);
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let multiplier = (1.0 + jitter_std * z).clamp(0.8, 1.2);
    base * multiplier
}

/// Send one HTTP request to the LLM and parse the response.
/// Returns Ok on any successful HTTP response (even if verdict is unparseable).
/// Returns Err only on HTTP/network failures.
pub async fn send_comparison_request(
    client: &Client,
    config: &LlmConfig,
    prompt: &str,
    narrow_win: f64,
) -> Result<(ParseResult, String, Option<Usage>, bool), String> {
    let request = ChatCompletionRequest {
        model: config.model.clone(),
        messages: vec![ChatMessage {
            role: "user",
            content: prompt.to_string(),
        }],
        temperature: jittered_temperature(config.temperature, config.temperature_jitter),
        max_tokens: config.max_tokens,
        logprobs: if config.logprobs { Some(true) } else { None },
        top_logprobs: if config.logprobs { Some(10) } else { None },
        presence_penalty: config.presence_penalty,
        top_p: config.top_p,
        stop: if config.logprobs {
            vec!["Verdict A:", "Verdict B:", "Verdict C:", "Verdict D:", "Verdict E:"]
        } else {
            vec![]
        },
        include_stop_str_in_output: if config.logprobs { Some(true) } else { None },
        reasoning: config.reasoning_effort.as_ref().map(|effort| ReasoningConfig {
            effort: effort.clone(),
        }),
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
        return Err(format!("LLM API returned {status}: {}", &body[..body.len().min(500)]));
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

    let parse_result = if config.logprobs {
        let logprobs = choice
            .logprobs
            .and_then(|lp| lp.content)
            .unwrap_or_default();

        if logprobs.is_empty() {
            crate::bail(format!("{} returned no logprobs. If your endpoint does not support logprobs, disable logprobs in your config.", config.model));
        }

        parse_response(&logprobs, narrow_win)
    } else {
        parse_response_text(&content, narrow_win)
    };

    Ok((parse_result, content, data.usage, hit_max_tokens))
}

/// Call the LLM to compare two items, with retries on HTTP errors.
///
/// Retries up to `max_retries` times with exponential backoff (1s, 4s, 16s).
/// Only HTTP/network errors trigger retries — unparseable verdicts do not.
pub async fn compare_pair(
    client: &Client,
    config: &LlmConfig,
    template: &str,
    criterion: &str,
    item1_name: &str,
    item2_name: &str,
    item1_id: i64,
    item2_id: i64,
    narrow_win: f64,
    analysis_length: &str,
    max_retries: usize,
    verbose: bool,
    judge_name: &str,
) -> Result<ComparisonResult, String> {
    let prompt = build_prompt(template, criterion, item1_name, item2_name, analysis_length);

    let mut last_err = String::new();
    for attempt in 0..=max_retries {
        match send_comparison_request(client, config, &prompt, narrow_win).await {
            Ok((parse_result, content, usage, hit_max_tokens)) => {
                return Ok(ComparisonResult {
                    item1_id,
                    item2_id,
                    parse_result,
                    response_text: content,
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
                            attempt + 1, max_retries, item1_name, item2_name, judge_name, last_err
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
        // With jitter_std = 0.0, should return base exactly
        assert_eq!(jittered_temperature(0.7, 0.0), 0.7);
        assert_eq!(jittered_temperature(1.0, 0.0), 1.0);
        assert_eq!(jittered_temperature(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_jittered_temperature_stays_in_range() {
        // With jitter, result should be within [base * 0.8, base * 1.2]
        let base = 0.7;
        for _ in 0..1000 {
            let result = jittered_temperature(base, 0.1);
            assert!(result >= base * 0.8, "result {result} < {}", base * 0.8);
            assert!(result <= base * 1.2, "result {result} > {}", base * 1.2);
        }
    }

    #[test]
    fn test_jittered_temperature_high_jitter_still_clamped() {
        // Even with extreme jitter, clamping should hold
        let base = 1.0;
        for _ in 0..1000 {
            let result = jittered_temperature(base, 10.0);
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
