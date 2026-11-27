//! `nanojudge probe`: send each configured judge a few cheap test requests and
//! append the findings, with the raw requests and responses, to the probe file.
//! The raw data is kept so new findings can be worked out from old probes.

use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde_json::{json, Value};

use crate::args::ProbeArgs;
use crate::config;
use crate::Error;
use crate::llm::{build_completions_url, describe_request_error, TOP_LOGPROBS};
use crate::resolve::{resolve_config, resolve_judges, ResolvedJudge, NO_DELIBERATION_MAX_TOKENS};

/// Bumped whenever the record layout or the set of test requests changes.
const PROBE_FORMAT_VERSION: u32 = 5;

/// Designed to test response, reasoning, and `max_tokens` enforcement in one
/// cheap request:
/// - `347*629` invites working things out, so a model that ignores
///   `reasoning_effort = "none"` either reports reasoning or burns the whole
///   16-token budget thinking with no visible answer.
/// - Repeating the 6-digit answer 20 times (~60-80 tokens) ensures the natural
///   reply overflows `max_tokens = 16`, catching endpoints that ignore the cap.
const PROBE_PROMPT: &str =
    "What is 347*629? Reply with only the number, repeated 20 times, one per line.";

/// Minimal prompt for the separate logprobs request when `judge.logprobs` is on.
/// Kept separate from `PROBE_PROMPT` so an endpoint that rejects `logprobs` or
/// `top_logprobs` with HTTP 400 still reports `responded`, `reasoned`, and
/// `max_tokens_respected` cleanly, and sent with `judge.max_tokens` so
/// reasoning models aren't starved of visible tokens.
const LOGPROBS_PROBE_PROMPT: &str = "Reply with only the letter A.";

/// Tokens a reply may use beyond one per byte of its visible answer, for
/// special tokens that aren't in the answer, like the end-of-turn token.
const PROBE_SPECIAL_TOKEN_ALLOWANCE: u64 = 5;


/// Where probe records are kept: `<data dir>/nanojudge/probes.jsonl`.
pub fn probes_path() -> crate::Result<PathBuf> {
    let data_dir = dirs::data_dir()
        .ok_or_else(|| Error::Io("could not find the user data directory".into()))?;
    Ok(data_dir.join("nanojudge").join("probes.jsonl"))
}

pub async fn run(args: ProbeArgs) -> crate::Result<()> {
    let (config_path, cfg) = config::load_config(args.config.as_deref())?;
    let resolved = resolve_config(&args.cfg, &cfg)?;
    let judges = resolve_judges(&args.cfg, &cfg, &config_path, resolved.deliberation_enabled, resolved.lineup_size)?;

    let path = probes_path()?;
    eprintln!("Probe records: {}", path.display());

    let client = Client::new();
    for judge in &judges {
        eprintln!("Probing {}", judge.display_name);
        let record = probe_judge(&client, judge).await?;
        append_record(&path, &record)?;
        let problems = judge_problems(judge, &record["findings"]);
        if problems.is_empty() {
            eprintln!("  OK: runs can use this judge");
        } else {
            for p in problems {
                eprintln!("  Runs will refuse this judge: {p}");
            }
        }
    }
    Ok(())
}

/// Check each judge against its latest probe, probing any judge that has
/// none. Pass only the judges that will send requests. Returns `Error::Probe`,
/// listing every problem, if a judge's probe failed or shows its settings
/// aren't honoured.
///
/// A probe belongs to a judge when the URL, model, `logprobs`,
/// `reasoning_effort`, `chat_template_kwargs` and `provider` match, and it's
/// in the current format. Other settings don't change what the probe tests,
/// except that a failed probe is redone under a longer `request_timeout_secs`
/// or larger `max_tokens` (see `reprobe_reason`).
pub async fn check_judges<'a>(judges: impl IntoIterator<Item = &'a ResolvedJudge>) -> crate::Result<()> {
    let path = probes_path()?;
    let records = read_records(&path)?;
    let client = Client::new();

    let mut problems = Vec::new();
    for judge in judges {
        let latest = latest_record(&records, judge);
        let record = match latest {
            Some(record) if reprobe_reason(record, judge).is_none() => record.clone(),
            Some(record) => {
                let reason = reprobe_reason(record, judge).expect("checked above");
                eprintln!(
                    "The last probe of {} failed under {reason}. Probing it again; records: {}",
                    judge.display_name, path.display()
                );
                let record = probe_judge(&client, judge).await?;
                append_record(&path, &record)?;
                record
            }
            None => {
                eprintln!("No probe of {} yet. Probing it now; records: {}", judge.display_name, path.display());
                let record = probe_judge(&client, judge).await?;
                append_record(&path, &record)?;
                record
            }
        };
        for p in judge_problems(judge, &record["findings"]) {
            problems.push(format!("{}: {p}", judge.display_name));
        }
    }

    if !problems.is_empty() {
        return Err(Error::Probe(format!(
            "the endpoint probe rules out these judges:\n  {}\nChange the judge settings, or fix the endpoint and run `nanojudge probe` to probe it again. Probe records: {}",
            problems.join("\n  "),
            path.display()
        )));
    }
    Ok(())
}

/// All probe records in the file, oldest first. Empty if the file doesn't
/// exist yet.
fn read_records(path: &PathBuf) -> crate::Result<Vec<Value>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::Io(format!("could not read {}: {e}", path.display()))),
    };
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            serde_json::from_str(line)
                .map_err(|e| Error::Input(format!("{} line {}: not valid JSON: {e}", path.display(), i + 1)))
        })
        .collect()
}

/// The most recent record that belongs to this judge, if any.
fn latest_record<'a>(records: &'a [Value], judge: &ResolvedJudge) -> Option<&'a Value> {
    let url = build_completions_url(&judge.endpoint);
    records.iter().rev().find(|r| {
        r["format_version"] == json!(PROBE_FORMAT_VERSION)
            && r["url"] == json!(url)
            && r["model"] == json!(judge.model)
            && r["settings"]["logprobs"] == json!(judge.logprobs)
            && r["settings"]["reasoning_effort"] == json!(judge.reasoning_effort)
            && r["settings"]["chat_template_kwargs"] == json!(judge.chat_template_kwargs)
            && r["settings"]["provider"] == json!(judge.provider)
    })
}

/// Why a judge's latest probe should be redone instead of reused, or `None` if
/// it can stand in. A passed probe is always reused. A failed one is redone
/// when `request_timeout_secs` is now longer (unset counts as longest) or
/// `max_tokens` is now larger, since tighter limits may be why it failed.
fn reprobe_reason(record: &Value, judge: &ResolvedJudge) -> Option<&'static str> {
    if judge_problems(judge, &record["findings"]).is_empty() {
        return None;
    }
    let old = record["settings"]["request_timeout_secs"].as_f64();
    let new = judge.request_timeout.map(|t| t.as_secs_f64());
    let longer = match (old, new) {
        (Some(old), Some(new)) => new > old,
        (Some(_), None) => true,
        (None, _) => false,
    };
    if longer {
        return Some("a shorter request_timeout_secs");
    }
    if record["settings"]["max_tokens"]
        .as_u64()
        .is_some_and(|old| u64::from(judge.max_tokens) > old)
    {
        return Some("a smaller max_tokens");
    }
    None
}

/// Why runs can't use this judge, given its probe findings. Empty if they can.
/// A finding the probe couldn't make ("can't tell") counts against the judge.
fn judge_problems(judge: &ResolvedJudge, findings: &Value) -> Vec<String> {
    let finding = |name: &str| findings[name].as_bool();

    if finding("responded") != Some(true) {
        return vec!["the endpoint gave no usable reply to the probe".into()];
    }

    let mut problems = Vec::new();
    match finding("max_tokens_respected") {
        Some(true) => {}
        Some(false) => problems.push("the endpoint doesn't keep to max_tokens".into()),
        None => problems.push("the probe couldn't tell whether the endpoint keeps to max_tokens".into()),
    }
    if judge.reasoning_effort.as_deref() == Some("none") {
        match finding("reasoned") {
            Some(false) => {}
            Some(true) => problems.push("the model reasons despite reasoning_effort = \"none\"".into()),
            None => problems.push("the probe couldn't tell whether the model reasons with reasoning_effort = \"none\"".into()),
        }
    }
    if judge.logprobs {
        match finding("logprobs_returned") {
            Some(true) => {}
            Some(false) => problems.push("logprobs are on but the endpoint returns none".into()),
            None => problems.push("logprobs are on but the probe couldn't tell whether the endpoint returns them".into()),
        }
        match findings["top_logprobs_returned"].as_u64() {
            Some(n) if n >= u64::from(TOP_LOGPROBS) => {}
            Some(n) => problems.push(format!(
                "logprobs are on but the endpoint returns only {n} top logprobs; runs need {TOP_LOGPROBS}"
            )),
            None => problems.push(format!(
                "logprobs are on but the probe couldn't tell whether the endpoint returns {TOP_LOGPROBS} top logprobs"
            )),
        }
    }
    problems
}

/// The request body shared by probe requests: the judge's own settings with
/// the given prompt and token limit.
fn base_request(judge: &ResolvedJudge, prompt: &str, max_tokens: u32) -> serde_json::Map<String, Value> {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(judge.model));
    body.insert("messages".into(), json!([{"role": "user", "content": prompt}]));
    body.insert("temperature".into(), json!(judge.temperature));
    body.insert("max_tokens".into(), json!(max_tokens));
    if let Some(p) = judge.presence_penalty {
        body.insert("presence_penalty".into(), json!(p));
    }
    if let Some(p) = judge.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if let Some(ref effort) = judge.reasoning_effort {
        body.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(ref kwargs) = judge.chat_template_kwargs {
        body.insert("chat_template_kwargs".into(), json!(kwargs));
    }
    if let Some(ref provider) = judge.provider {
        body.insert("provider".into(), json!(provider));
    }
    body
}

async fn probe_judge(client: &Client, judge: &ResolvedJudge) -> crate::Result<Value> {
    let url = build_completions_url(&judge.endpoint);
    let max_tokens_limit = judge.max_tokens.min(NO_DELIBERATION_MAX_TOKENS);

    let main_body = base_request(judge, PROBE_PROMPT, max_tokens_limit);
    let mut requests = vec![send_test(client, judge, &url, "probe", Value::Object(main_body)).await];

    let responded = first_choice(&requests[0]["response"]).is_some();
    let did_reason = reasoned(&requests[0]["response"]);
    let kept_max_tokens = max_tokens_respected(&requests[0]["response"], max_tokens_limit);
    let (has_logprobs, top_lp_count) = if judge.logprobs && responded {
        let mut lp_body = base_request(judge, LOGPROBS_PROBE_PROMPT, judge.max_tokens);
        lp_body.insert("logprobs".into(), json!(true));
        lp_body.insert("top_logprobs".into(), json!(TOP_LOGPROBS));
        requests.push(send_test(client, judge, &url, "logprobs", Value::Object(lp_body)).await);
        (
            logprobs_returned(&requests[1]["response"]),
            top_logprobs_returned(&requests[1]["response"]),
        )
    } else {
        (None, None)
    };

    eprintln!("  responded: {}", finding_str(Some(responded)));
    eprintln!("  reasoned: {}", finding_str(did_reason));
    if judge.logprobs {
        eprintln!("  logprobs returned: {}", finding_str(has_logprobs));
        match top_lp_count {
            Some(n) => eprintln!("  top logprobs returned: {n} of {TOP_LOGPROBS}"),
            None => eprintln!("  top logprobs returned: can't tell"),
        }
    }
    eprintln!("  max_tokens respected: {}", finding_str(kept_max_tokens));

    Ok(json!({
        "format_version": PROBE_FORMAT_VERSION,
        "probed_at_unix_ms": unix_ms()?,
        "nanojudge_version": env!("CARGO_PKG_VERSION"),
        "endpoint": judge.endpoint,
        "url": url,
        "model": judge.model,
        "settings": {
            "logprobs": judge.logprobs,
            "reasoning_effort": judge.reasoning_effort,
            "chat_template_kwargs": judge.chat_template_kwargs,
            "provider": judge.provider,
            "max_tokens": judge.max_tokens,
            "request_timeout_secs": judge.request_timeout.map(|t| t.as_secs_f64()),
        },
        "findings": {
            "responded": responded,
            "reasoned": did_reason,
            "logprobs_returned": has_logprobs,
            "top_logprobs_returned": top_lp_count,
            "max_tokens_respected": kept_max_tokens,
        },
        "requests": requests,
    }))
}

fn finding_str(finding: Option<bool>) -> &'static str {
    match finding {
        Some(true) => "yes",
        Some(false) => "no",
        None => "can't tell",
    }
}

/// The first choice of a reply. None if the response isn't a usable reply
/// (an HTTP error body, a failed request, text that isn't JSON).
fn first_choice(response: &Value) -> Option<&Value> {
    let choice = response.get("choices")?.get(0)?;
    choice.get("message")?;
    Some(choice)
}

/// The logprob entries of a reply's first choice. Empty if there are none.
fn logprob_entries(choice: &Value) -> &[Value] {
    choice
        .pointer("/logprobs/content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// Whether a reply to a request with `logprobs: true` has logprobs.
/// None if the response isn't a usable reply.
fn logprobs_returned(response: &Value) -> Option<bool> {
    let choice = first_choice(response)?;
    Some(!logprob_entries(choice).is_empty())
}

/// How many top logprobs came back per token: the smallest count across the
/// reply's tokens. None if the response isn't a usable reply or has no
/// logprobs.
fn top_logprobs_returned(response: &Value) -> Option<usize> {
    let choice = first_choice(response)?;
    logprob_entries(choice)
        .iter()
        .map(|t| t.get("top_logprobs").and_then(Value::as_array).map_or(0, Vec::len))
        .min()
}

/// Whether the reply kept to `limit`.
/// None if the response isn't a usable reply or reports no token count.
fn max_tokens_respected(response: &Value, limit: u32) -> Option<bool> {
    first_choice(response)?;
    let completion_tokens = response.pointer("/usage/completion_tokens").and_then(Value::as_u64)?;
    Some(completion_tokens <= u64::from(limit))
}

/// Whether a reply shows the model reasoned.
///
/// Some(true) on any sign of reasoning: reasoning text, reported reasoning
/// tokens, `<think>` tags in the content, or more completion tokens than the
/// visible answer can account for. Every token of the answer is at least one
/// byte, so beyond its byte length (plus a few special tokens) the extra tokens
/// are reasoning the endpoint counted but hid from the reply. Some(false) when
/// the completion tokens fit in the answer. None if the response isn't a usable
/// reply or reports no token count.
///
/// Hidden reasoning shorter than the answer's spare bytes goes unnoticed,
/// though a 3-digit multiplication under a 16-token cap leaves a reasoning
/// model with no visible answer at all.
fn reasoned(response: &Value) -> Option<bool> {
    let choice = first_choice(response)?;
    let message = &choice["message"];

    let has_text = |field: &str| message.get(field).and_then(Value::as_str).is_some_and(|s| !s.is_empty());
    if has_text("reasoning") || has_text("reasoning_content") {
        return Some(true);
    }
    let reasoning_tokens = response
        .pointer("/usage/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64);
    if reasoning_tokens.is_some_and(|n| n > 0) {
        return Some(true);
    }
    let content = message.get("content").and_then(Value::as_str).unwrap_or("");
    if content.contains("<think>") || content.contains("</think>") {
        return Some(true);
    }

    let completion_tokens = response.pointer("/usage/completion_tokens").and_then(Value::as_u64)?;
    Some(completion_tokens > content.len() as u64 + PROBE_SPECIAL_TOKEN_ALLOWANCE)
}

/// Send one test request and record it. Never records headers, so the API key
/// stays out of the probe file.
async fn send_test(client: &Client, judge: &ResolvedJudge, url: &str, test: &str, body: Value) -> Value {
    let mut req = client.post(url).json(&body);
    if let Some(ref key) = judge.api_key {
        req = req.bearer_auth(key);
    }
    if let Some(timeout) = judge.request_timeout {
        req = req.timeout(timeout);
    }

    let start = Instant::now();
    let (http_status, response, error) = match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                // Keep the response as JSON when it is JSON, as text otherwise.
                Ok(text) => {
                    let response = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text));
                    (Some(status), Some(response), None)
                }
                Err(e) => (Some(status), None, Some(format!(
                    "reading the response failed: {}", describe_request_error(&e, judge.request_timeout),
                ))),
            }
        }
        Err(e) => (None, None, Some(format!(
            "the request failed: {}", describe_request_error(&e, judge.request_timeout),
        ))),
    };
    let elapsed_ms = start.elapsed().as_millis() as u64;

    match (&http_status, &error) {
        (_, Some(e)) => eprintln!("  {test}: {e}"),
        (Some(s), None) => eprintln!("  {test}: HTTP {s} in {elapsed_ms} ms"),
        (None, None) => unreachable!("a request with no status always has an error"),
    }

    json!({
        "test": test,
        "request": body,
        "http_status": http_status,
        "response": response,
        "error": error,
        "elapsed_ms": elapsed_ms,
    })
}

fn unix_ms() -> crate::Result<u64> {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::Io(format!("system clock is before 1970: {e}")))?;
    Ok(dur.as_millis() as u64)
}

/// Append one record as a single line, in a single write.
fn append_record(path: &PathBuf, record: &Value) -> crate::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| Error::Io(format!("could not create {}: {e}", dir.display())))?;
    }
    let mut line = serde_json::to_string(record)
        .expect("probe record Value always serializes to JSON");
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| Error::Io(format!("could not open {}: {e}", path.display())))?;
    file.write_all(line.as_bytes())
        .map_err(|e| Error::Io(format!("could not write to {}: {e}", path.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(message: Value, tokens: Option<&[&str]>) -> Value {
        let logprobs = tokens.map(|t| json!({"content": t.iter().map(|tok| json!({"token": tok})).collect::<Vec<_>>()}));
        json!({"choices": [{"message": message, "logprobs": logprobs}]})
    }

    fn reply_with_tokens(message: Value, completion_tokens: u64) -> Value {
        json!({"choices": [{"message": message}], "usage": {"completion_tokens": completion_tokens}})
    }

    #[test]
    fn reasoned_when_reasoning_text_is_present() {
        let r = reply_with_tokens(json!({"content": "935", "reasoning_content": "55*17 is..."}), 3);
        assert_eq!(reasoned(&r), Some(true));
        let r = reply_with_tokens(json!({"content": "935", "reasoning": "55*17 is..."}), 3);
        assert_eq!(reasoned(&r), Some(true));
    }

    #[test]
    fn reasoned_when_reasoning_tokens_are_reported() {
        let mut r = reply_with_tokens(json!({"content": "935"}), 3);
        r["usage"]["completion_tokens_details"] = json!({"reasoning_tokens": 1});
        assert_eq!(reasoned(&r), Some(true));
    }

    #[test]
    fn reasoned_when_content_has_think_tags() {
        let r = reply_with_tokens(json!({"content": "<think>\nhmm</think>\n\n935"}), 9);
        assert_eq!(reasoned(&r), Some(true));
    }

    #[test]
    fn reasoned_when_tokens_exceed_the_visible_answer() {
        // "935" is 3 bytes, so at most 3 + 5 tokens without hidden reasoning.
        let r = reply_with_tokens(json!({"content": "935", "reasoning": null}), 9);
        assert_eq!(reasoned(&r), Some(true));
        // Cut off mid-reasoning: no answer at all.
        let r = reply_with_tokens(json!({"content": null}), 16);
        assert_eq!(reasoned(&r), Some(true));
    }

    #[test]
    fn not_reasoned_when_tokens_fit_in_the_visible_answer() {
        let r = reply_with_tokens(json!({"content": "935"}), 8);
        assert_eq!(reasoned(&r), Some(false));
        let r = reply_with_tokens(json!({"content": "935", "reasoning": ""}), 2);
        assert_eq!(reasoned(&r), Some(false));
    }

    #[test]
    fn cant_tell_without_token_counts_or_a_usable_reply() {
        assert_eq!(reasoned(&reply(json!({"content": "935"}), None)), None);
        assert_eq!(reasoned(&Value::Null), None);
        assert_eq!(reasoned(&json!("Bad Gateway")), None);
        assert_eq!(reasoned(&json!({"error": {"message": "nope"}})), None);
    }

    fn judge(reasoning_effort: Option<&str>, logprobs: bool) -> ResolvedJudge {
        ResolvedJudge {
            endpoint: "http://127.0.0.1:8001".into(),
            model: "m".into(),
            api_key: None,
            temperature: 0.7,
            temperature_jitter: 0.0,
            presence_penalty: None,
            top_p: None,
            logprobs,
            concurrency: 1,
            weight: 1.0,
            min_logprob_coverage: 0.0,
            verdict_temperature: 1.0,
            max_tokens: 16,
            reasoning_effort: reasoning_effort.map(String::from),
            chat_template_kwargs: None,
            provider: None,
            request_timeout: None,
            judge_id: 0,
            display_name: "m".into(),
        }
    }

    fn good_findings() -> Value {
        json!({
            "responded": true,
            "reasoned": false,
            "logprobs_returned": true,
            "top_logprobs_returned": 20,
            "max_tokens_respected": true,
        })
    }

    fn record(url: &str, model: &str, logprobs: bool, effort: Option<&str>, kwargs: Value, tag: u32) -> Value {
        json!({
            "format_version": PROBE_FORMAT_VERSION,
            "url": url,
            "model": model,
            "settings": {
                "logprobs": logprobs,
                "reasoning_effort": effort,
                "chat_template_kwargs": kwargs,
                "provider": null,
            },
            "tag": tag,
        })
    }

    #[test]
    fn latest_record_matches_url_model_and_request_settings() {
        let url = "http://127.0.0.1:8001/v1/chat/completions";
        let mut j = judge(Some("none"), false);
        let records = vec![
            record(url, "m", false, Some("none"), Value::Null, 1),
            record(url, "m", false, Some("none"), Value::Null, 2),
            record(url, "other", false, Some("none"), Value::Null, 3),
            record(url, "m", false, None, Value::Null, 4),
            record("http://elsewhere/v1/chat/completions", "m", false, Some("none"), Value::Null, 5),
            record(url, "m", false, Some("none"), json!({"enable_thinking": false}), 6),
            record(url, "m", true, Some("none"), Value::Null, 7),
        ];
        assert_eq!(latest_record(&records, &j).unwrap()["tag"], 2);

        // A trailing slash on the endpoint builds the same URL.
        j.endpoint = "http://127.0.0.1:8001/".into();
        assert_eq!(latest_record(&records, &j).unwrap()["tag"], 2);

        j.logprobs = true;
        assert_eq!(latest_record(&records, &j).unwrap()["tag"], 7);

        j.logprobs = false;
        j.chat_template_kwargs = Some([("enable_thinking".to_string(), json!(false))].into());
        assert_eq!(latest_record(&records, &j).unwrap()["tag"], 6);

        j.chat_template_kwargs = None;
        j.reasoning_effort = Some("low".into());
        assert!(latest_record(&records, &j).is_none());
    }

    #[test]
    fn failed_probe_is_redone_under_a_longer_timeout_or_larger_max_tokens() {
        use std::time::Duration;
        let url = "http://127.0.0.1:8001/v1/chat/completions";
        let mut passed = record(url, "m", false, None, Value::Null, 1);
        passed["findings"] = good_findings();
        let mut failed = record(url, "m", false, None, Value::Null, 2);
        failed["findings"] = json!({"responded": false});
        failed["settings"]["request_timeout_secs"] = json!(5);
        failed["settings"]["max_tokens"] = json!(64);
        let mut failed_unset = record(url, "m", false, None, Value::Null, 3);
        failed_unset["findings"] = json!({"responded": false});

        let mut j = judge(None, false);
        j.max_tokens = 64;
        for (timeout, failed_reusable) in [(Some(1.0), true), (Some(5.0), true), (Some(5.5), false), (None, false)] {
            j.request_timeout = timeout.map(Duration::from_secs_f64);
            assert!(reprobe_reason(&passed, &j).is_none());
            assert_eq!(reprobe_reason(&failed, &j).is_none(), failed_reusable, "timeout {timeout:?}");
            // A failed probe with no timeout already waited indefinitely.
            assert!(reprobe_reason(&failed_unset, &j).is_none());
        }

        // Larger max_tokens re-probes a failed record, never a passed one.
        j.request_timeout = Some(Duration::from_secs(5));
        j.max_tokens = 128;
        assert!(reprobe_reason(&passed, &j).is_none());
        assert_eq!(reprobe_reason(&failed, &j), Some("a smaller max_tokens"));
    }

    #[test]
    fn latest_record_matches_provider() {
        let url = "https://openrouter.ai/api/v1/chat/completions";
        let pinned = json!({"only": ["xiaomi"], "allow_fallbacks": false});
        let mut other = record(url, "m", false, None, Value::Null, 1);
        other["settings"]["provider"] = json!({"only": ["novita"], "allow_fallbacks": false});
        let mut same = record(url, "m", false, None, Value::Null, 2);
        same["settings"]["provider"] = pinned.clone();
        let records = vec![same, other];

        let mut j = judge(None, false);
        j.endpoint = "https://openrouter.ai/api/v1".into();
        j.provider = Some(serde_json::from_value(pinned).unwrap());
        assert_eq!(latest_record(&records, &j).unwrap()["tag"], 2);

        j.provider = Some(serde_json::from_value(json!({"only": ["deepinfra"]})).unwrap());
        assert!(latest_record(&records, &j).is_none());
    }

    #[test]
    fn latest_record_skips_other_formats() {
        let url = "http://127.0.0.1:8001/v1/chat/completions";
        let mut old = record(url, "m", false, None, Value::Null, 1);
        old["format_version"] = json!(PROBE_FORMAT_VERSION - 1);
        assert!(latest_record(&[old], &judge(None, false)).is_none());
    }

    #[test]
    fn good_findings_allow_every_setting() {
        assert!(judge_problems(&judge(Some("none"), true), &good_findings()).is_empty());
    }

    #[test]
    fn no_reply_is_the_only_problem_reported() {
        let mut f = good_findings();
        f["responded"] = json!(false);
        f["max_tokens_respected"] = Value::Null;
        assert_eq!(judge_problems(&judge(Some("none"), true), &f).len(), 1);
    }

    #[test]
    fn max_tokens_must_be_respected_by_every_judge() {
        for value in [json!(false), Value::Null] {
            let mut f = good_findings();
            f["max_tokens_respected"] = value;
            assert_eq!(judge_problems(&judge(None, false), &f).len(), 1);
        }
    }

    #[test]
    fn reasoning_is_only_checked_with_reasoning_effort_none() {
        for value in [json!(true), Value::Null] {
            let mut f = good_findings();
            f["reasoned"] = value;
            assert_eq!(judge_problems(&judge(Some("none"), false), &f).len(), 1);
            assert!(judge_problems(&judge(None, false), &f).is_empty());
            assert!(judge_problems(&judge(Some("high"), false), &f).is_empty());
        }
    }

    #[test]
    fn logprobs_are_only_checked_in_logprobs_mode() {
        for (name, value) in [
            ("logprobs_returned", json!(false)),
            ("logprobs_returned", Value::Null),
            ("top_logprobs_returned", json!(TOP_LOGPROBS - 1)),
            ("top_logprobs_returned", Value::Null),
        ] {
            let mut f = good_findings();
            f[name] = value;
            assert_eq!(judge_problems(&judge(None, true), &f).len(), 1, "{name}");
            assert!(judge_problems(&judge(None, false), &f).is_empty(), "{name}");
        }
        let mut f = good_findings();
        f["top_logprobs_returned"] = json!(TOP_LOGPROBS);
        assert!(judge_problems(&judge(None, true), &f).is_empty());
    }

    #[test]
    fn logprobs_returned_checks_for_entries() {
        assert_eq!(logprobs_returned(&reply(json!({"content": "935"}), Some(&["935"]))), Some(true));
        assert_eq!(logprobs_returned(&reply(json!({"content": "935"}), Some(&[]))), Some(false));
        assert_eq!(logprobs_returned(&reply(json!({"content": "935"}), None)), Some(false));
        assert_eq!(logprobs_returned(&json!({"error": {"message": "nope"}})), None);
    }

    #[test]
    fn top_logprobs_returned_takes_the_smallest_count() {
        let alts = |n: usize| (0..n).map(|i| json!({"token": i.to_string()})).collect::<Vec<_>>();
        let r = json!({"choices": [{"message": {"content": "ab"}, "logprobs": {"content": [
            {"token": "a", "top_logprobs": alts(20)},
            {"token": "b", "top_logprobs": alts(18)},
        ]}}]});
        assert_eq!(top_logprobs_returned(&r), Some(18));
        // Entries without top_logprobs count as 0.
        assert_eq!(top_logprobs_returned(&reply(json!({"content": "a"}), Some(&["a"]))), Some(0));
        assert_eq!(top_logprobs_returned(&reply(json!({"content": "a"}), None)), None);
        assert_eq!(top_logprobs_returned(&json!({"error": {"message": "nope"}})), None);
    }

    #[test]
    fn max_tokens_respected_compares_completion_tokens() {
        let with_usage = |n: u64| json!({"choices": [{"message": {"content": null}}], "usage": {"completion_tokens": n}});
        assert_eq!(max_tokens_respected(&with_usage(16), 16), Some(true));
        assert_eq!(max_tokens_respected(&with_usage(17), 16), Some(false));
        assert_eq!(max_tokens_respected(&with_usage(8), 8), Some(true));
        assert_eq!(max_tokens_respected(&with_usage(9), 8), Some(false));
        assert_eq!(max_tokens_respected(&reply(json!({"content": "935"}), None), 16), None);
        assert_eq!(max_tokens_respected(&json!({"error": {"message": "nope"}}), 16), None);
    }
}
