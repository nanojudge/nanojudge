/// Verdict extraction for pairwise comparisons.
///
/// Two separate parsing modes:
/// - Logprob mode: extracts continuous probabilities from token logprobs.
/// - Text mode (--no-logprobs): extracts discrete verdict letter from response text.
use serde::Deserialize;

/// Default narrow-win probability (B and D on the likert scale).
pub const DEFAULT_NARROW_WIN: f64 = 0.8;

/// Default minimum fraction of A-E probability mass the top-logprobs must
/// cover before the logprob-derived probability is trusted. Below this, the
/// comparison parse fails (returns None).
pub const DEFAULT_MIN_LOGPROB_COVERAGE: f64 = 0.95;

/// Build likert mapping from a narrow-win value.
/// A=1.0, B=narrow_win, C=0.5, D=1.0-narrow_win, E=0.0.
fn likert_mapping(narrow_win: f64) -> [f64; 5] {
    [1.0, narrow_win, 0.5, 1.0 - narrow_win, 0.0]
}

/// The 5 likert letters in order.
const LIKERT_LETTERS: [char; 5] = ['A', 'B', 'C', 'D', 'E'];

/// A single top-logprob entry from the OpenAI response.
#[derive(Debug, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
}

/// A single token's logprob info from the OpenAI response.
#[derive(Debug, Deserialize)]
pub struct LogprobContent {
    pub token: String,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

/// Result of parsing a comparison response.
pub struct ParseResult {
    /// P(item1 wins), 0.0 to 1.0. None if extraction failed.
    pub item1_win_probability: Option<f64>,
}

fn letter_to_index(c: char) -> Option<usize> {
    LIKERT_LETTERS.iter().position(|&l| l == c.to_ascii_uppercase())
}

/// Extract likert choice probabilities from logprobs.
///
/// Returns (choice_probs, expected_p1) or (None, None) if extraction fails.
fn extract_likert_probabilities(logprobs: &[LogprobContent], mapping: &[f64; 5], min_logprob_coverage: f64) -> (Option<[f64; 5]>, Option<f64>) {
    if logprobs.is_empty() {
        return (None, None);
    }

    let tokens: Vec<&str> = logprobs.iter().map(|lp| lp.token.as_str()).collect();

    // Find "Verdict" marker in logprob tokens
    let mut search_start = 0;
    for (i, raw_tok) in tokens.iter().enumerate() {
        let t = raw_tok.trim().to_lowercase();
        if t.starts_with("verdict") {
            search_start = i + 1;
            break;
        }
        if (t == "ver" || t == "verd") && i + 1 < tokens.len() {
            let next_t = tokens[i + 1].trim().to_lowercase();
            if next_t == "dict" || next_t == "dict:" || next_t == "ict" || next_t == "ict:" {
                search_start = i + 2;
                break;
            }
        }
    }

    let search_end = (search_start + 10).min(tokens.len());

    for i in search_start..search_end {
        let tok = tokens[i].trim();
        if tok.is_empty() {
            continue;
        }

        let first_char = tok.chars().next().unwrap();
        if letter_to_index(first_char).is_none() {
            continue;
        }

        let rest = &tok[first_char.len_utf8()..];
        if !rest.is_empty() && rest != ":" {
            continue;
        }

        // Skip lowercase 'a' without colon (likely the word "a", not choice A)
        let has_colon = tok.contains(':');
        let is_upper = first_char.is_uppercase();
        if !is_upper && !has_colon && first_char.to_ascii_lowercase() == 'a' {
            continue;
        }

        let top_logprobs = match &logprobs[i].top_logprobs {
            Some(tlps) if !tlps.is_empty() => tlps,
            _ => return (None, None),
        };

        let mut choice_probs = [0.0_f64; 5];

        for tlp in top_logprobs {
            let clean = tlp.token.trim().trim_end_matches(':');
            if clean.len() == 1 {
                if let Some(tidx) = letter_to_index(clean.chars().next().unwrap()) {
                    choice_probs[tidx] += tlp.logprob.exp();
                }
            }
        }

        let prob_sum: f64 = choice_probs.iter().sum();
        if prob_sum >= min_logprob_coverage {
            // Normalize
            for p in &mut choice_probs {
                *p /= prob_sum;
            }
            let expected_p1: f64 = choice_probs
                .iter()
                .zip(mapping.iter())
                .map(|(p, m)| p * m)
                .sum();
            return (Some(choice_probs), Some(expected_p1));
        } else {
            // Logprobs don't cover enough of the A-E space — fall through to text
            return (None, None);
        }
    }

    (None, None)
}

/// Parse a comparison response into a win probability.
///
/// Logprobs only — no text fallback. Returns None if logprob extraction fails.
pub fn parse_response(logprobs: &[LogprobContent], narrow_win: f64, min_logprob_coverage: f64) -> ParseResult {
    let mapping = likert_mapping(narrow_win);
    let (_, expected_p1) = extract_likert_probabilities(logprobs, &mapping, min_logprob_coverage);

    ParseResult {
        item1_win_probability: expected_p1,
    }
}

/// Parse a verdict letter from response text (for --no-logprobs mode).
///
/// Finds the last "Verdict [A-E]" in the text and maps the letter to a
/// win probability using the likert scale. Uses the last occurrence to
/// handle cases where "verdict" appears in the analysis before the final verdict.
pub fn parse_response_text(text: &str, narrow_win: f64) -> ParseResult {
    let mapping = likert_mapping(narrow_win);
    // Use ASCII lowercase so byte offsets stay aligned with the original text —
    // we need that alignment below to check the original character's case.
    let lower = text.to_ascii_lowercase();
    let orig_bytes = text.as_bytes();
    let mut result = None;
    let mut search_start = 0;

    while let Some(offset) = lower[search_start..].find("verdict") {
        let after_verdict = search_start + offset + 7; // len("verdict")
        let mut saw_colon = false;
        for (byte_off, c) in lower[after_verdict..].char_indices() {
            match c {
                ' ' | '\t' | '\n' | '\r' | '*' | '#' => continue,
                ':' => {
                    saw_colon = true;
                    continue;
                }
                _ => {
                    if let Some(idx) = letter_to_index(c) {
                        // Mirror the logprob-mode guard: a bare lowercase 'a'
                        // with no preceding colon is almost certainly the
                        // English article ("the verdict a reader reaches"),
                        // not a Verdict A answer. Uppercase 'A' is always a
                        // verdict; lowercase 'a' only counts after a colon.
                        let is_lower_a = c == 'a' && orig_bytes[after_verdict + byte_off] != b'A';
                        if is_lower_a && !saw_colon {
                            break;
                        }
                        result = Some(mapping[idx]);
                    }
                    break;
                }
            }
        }
        search_start = after_verdict;
    }

    ParseResult {
        item1_win_probability: result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letter_to_index() {
        assert_eq!(letter_to_index('A'), Some(0));
        assert_eq!(letter_to_index('a'), Some(0));
        assert_eq!(letter_to_index('E'), Some(4));
        assert_eq!(letter_to_index('F'), None);
    }

    #[test]
    fn test_extract_likert_from_logprobs() {
        // Simulate logprobs where the model outputs "Verdict:" then "B"
        // with top_logprobs showing strong preference for B
        let logprobs = vec![
            LogprobContent {
                token: "Verdict".to_string(),
                top_logprobs: None,
            },
            LogprobContent {
                token: ":".to_string(),
                top_logprobs: None,
            },
            LogprobContent {
                token: " ".to_string(),
                top_logprobs: None,
            },
            LogprobContent {
                token: "B".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "B".to_string(), logprob: -0.05 },   // ~0.951
                    TopLogprob { token: "A".to_string(), logprob: -3.5 },    // ~0.030
                    TopLogprob { token: "C".to_string(), logprob: -4.5 },    // ~0.011
                    TopLogprob { token: "D".to_string(), logprob: -6.0 },    // ~0.002
                    TopLogprob { token: "E".to_string(), logprob: -7.0 },    // ~0.001
                ]),
            },
        ];

        let mapping = likert_mapping(DEFAULT_NARROW_WIN);
        let (choice_probs, expected_p1) = extract_likert_probabilities(&logprobs, &mapping, DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(choice_probs.is_some(), "choice_probs should be Some");
        assert!(expected_p1.is_some(), "expected_p1 should be Some");

        let p1 = expected_p1.unwrap();
        // B is dominant (0.8 mapping), so expected_p1 should be close to 0.8
        assert!(p1 > 0.7, "expected_p1 {p1} should be > 0.7");
        assert!(p1 < 0.9, "expected_p1 {p1} should be < 0.9");
    }

    #[test]
    fn test_parse_response_with_logprobs() {
        let logprobs = vec![
            LogprobContent {
                token: "Verdict".to_string(),
                top_logprobs: None,
            },
            LogprobContent {
                token: ":".to_string(),
                top_logprobs: None,
            },
            LogprobContent {
                token: "\n".to_string(),
                top_logprobs: None,
            },
            LogprobContent {
                token: "B".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "B".to_string(), logprob: -0.05 },
                    TopLogprob { token: "A".to_string(), logprob: -3.5 },
                    TopLogprob { token: "C".to_string(), logprob: -4.0 },
                    TopLogprob { token: "D".to_string(), logprob: -5.0 },
                    TopLogprob { token: "E".to_string(), logprob: -6.0 },
                ]),
            },
        ];

        let result = parse_response(&logprobs, DEFAULT_NARROW_WIN, DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(result.item1_win_probability.is_some());
        let p = result.item1_win_probability.unwrap();
        assert!(p > 0.7);
    }

    #[test]
    fn test_min_logprob_coverage_gates_parsing() {
        // A-E mass sums to 0.96 (the rest is on a non-letter token).
        let logprobs = vec![
            LogprobContent { token: "Verdict".to_string(), top_logprobs: None },
            LogprobContent { token: ":".to_string(), top_logprobs: None },
            LogprobContent {
                token: "B".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "B".to_string(), logprob: (0.96_f64).ln() },
                    TopLogprob { token: " the".to_string(), logprob: (0.04_f64).ln() },
                ]),
            },
        ];

        // 0.96 coverage clears the 0.95 threshold.
        let result = parse_response(&logprobs, DEFAULT_NARROW_WIN, 0.95);
        assert!(result.item1_win_probability.is_some());

        // ...but not a stricter 0.99 threshold.
        let result = parse_response(&logprobs, DEFAULT_NARROW_WIN, 0.99);
        assert!(result.item1_win_probability.is_none());
    }

    #[test]
    fn test_parse_response_no_logprobs_returns_none() {
        let result = parse_response(&[], DEFAULT_NARROW_WIN, DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(result.item1_win_probability.is_none());
    }

    #[test]
    fn test_parse_response_unparseable() {
        let result = parse_response(&[], DEFAULT_NARROW_WIN, DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(result.item1_win_probability.is_none());
    }

    #[test]
    fn test_custom_narrow_win_no_logprobs_returns_none() {
        let result = parse_response(&[], 0.7, DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(result.item1_win_probability.is_none());

        let result = parse_response(&[], 0.7, DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(result.item1_win_probability.is_none());
    }

    // --- Text-based parsing tests (--no-logprobs mode) ---

    #[test]
    fn test_text_parse_verdict_b() {
        let text = "Some analysis.\n\nVerdict B: Option 1, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.8));
    }

    #[test]
    fn test_text_parse_verdict_a() {
        let text = "Analysis here.\n\nVerdict A: Option 1, clearly";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(1.0));
    }

    #[test]
    fn test_text_parse_verdict_e() {
        let text = "Analysis here.\n\nVerdict E: Option 2, clearly";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.0));
    }

    #[test]
    fn test_text_parse_verdict_c_draw() {
        let text = "Analysis here.\n\nVerdict C: Too close to call";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.5));
    }

    #[test]
    fn test_text_parse_verdict_d_with_custom_narrow_win() {
        let text = "Analysis.\n\nVerdict D: Option 2, marginally";
        let result = parse_response_text(text, 0.7);
        let p = result.item1_win_probability.unwrap();
        assert!((p - 0.3).abs() < 1e-10, "expected ~0.3, got {p}"); // 1.0 - 0.7
    }

    #[test]
    fn test_text_parse_verdict_with_colon_separator() {
        let text = "Analysis.\n\nVerdict: B: Option 1, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.8));
    }

    #[test]
    fn test_text_parse_uses_last_verdict() {
        // "verdict" appears in the analysis, but we want the final one
        let text = "The verdict on flavor is mixed.\n\nVerdict D: Option 2, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        let p = result.item1_win_probability.unwrap();
        assert!((p - 0.2).abs() < 1e-10, "expected ~0.2, got {p}");
    }

    #[test]
    fn test_text_parse_no_verdict_returns_none() {
        let text = "Some analysis without a final answer.";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert!(result.item1_win_probability.is_none());
    }

    #[test]
    fn test_text_parse_case_insensitive() {
        let text = "Analysis.\n\nVERDICT B: Option 1, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.8));
    }

    #[test]
    fn test_text_parse_lowercase_verdict_letter() {
        let text = "Analysis.\n\nVerdict b: Option 1, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.8));
    }

    #[test]
    fn test_text_parse_bold_wrapped_verdict() {
        let text = "Analysis.\n\n**Verdict E**";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.0));
    }

    #[test]
    fn test_text_parse_bold_verdict_with_colon() {
        let text = "Analysis.\n\n**Verdict B:** Option 1, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(0.8));
    }

    #[test]
    fn test_text_parse_bold_only_keyword() {
        let text = "Analysis.\n\n**Verdict** A";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(1.0));
    }

    #[test]
    fn test_text_parse_heading_verdict() {
        let text = "Analysis.\n\n## Verdict D: Option 2, marginally";
        let result = parse_response_text(text, DEFAULT_NARROW_WIN);
        assert_eq!(result.item1_win_probability, Some(1.0 - DEFAULT_NARROW_WIN));
    }
}
