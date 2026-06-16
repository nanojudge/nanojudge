/// Verdict extraction for pairwise comparisons.
///
/// Two separate parsing modes:
/// - Logprob mode: extracts continuous probabilities from token logprobs.
/// - Text mode (--no-logprobs): extracts discrete verdict letter from response text.
use serde::Deserialize;

/// Default minimum fraction of A-D probability mass the top-logprobs must
/// cover before the logprob-derived distribution is trusted. Below this, the
/// comparison parse fails (returns None).
pub const DEFAULT_MIN_LOGPROB_COVERAGE: f64 = 0.95;

/// The 4 verdict letters in order (A clear win … D clear loss).
const LIKERT_LETTERS: [char; 4] = ['A', 'B', 'C', 'D'];

/// One-hot category distribution for a single chosen letter index.
fn one_hot(idx: usize) -> [f64; 4] {
    let mut v = [0.0; 4];
    v[idx] = 1.0;
    v
}

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
    /// Categorical verdict distribution `[P(A),P(B),P(C),P(D)]` from item1's
    /// perspective. None if extraction failed.
    pub category_probs: Option<[f64; 4]>,
}

fn letter_to_index(c: char) -> Option<usize> {
    LIKERT_LETTERS.iter().position(|&l| l == c.to_ascii_uppercase())
}

/// Extract the judge's categorical verdict distribution `[P(A)..P(D)]` from
/// logprobs. Returns None if no "Verdict" marker is present, extraction fails,
/// or the A-D mass is below the coverage threshold.
fn extract_likert_probabilities(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> Option<[f64; 4]> {
    if logprobs.is_empty() {
        return None;
    }

    let tokens: Vec<&str> = logprobs.iter().map(|lp| lp.token.as_str()).collect();

    // Collect every "Verdict" marker position. The analysis prose can contain
    // the word "verdict" (e.g. "reach a fair verdict") before or after the
    // real "Verdict: x" line, so we attempt extraction after each marker and
    // keep the LAST successful one. Mirrors parse_response_text, which keeps
    // the last marker that actually yields a letter.
    // No marker at all means the judge ignored the required "Verdict: x"
    // format — discard rather than scan the response start for stray letters.
    let mut marker_starts = Vec::new();
    for (i, raw_tok) in tokens.iter().enumerate() {
        let t = raw_tok.trim().to_lowercase();
        if t.starts_with("verdict") {
            marker_starts.push(i + 1);
            continue;
        }
        if (t == "ver" || t == "verd") && i + 1 < tokens.len() {
            let next_t = tokens[i + 1].trim().to_lowercase();
            if next_t == "dict" || next_t == "dict:" || next_t == "ict" || next_t == "ict:" {
                marker_starts.push(i + 2);
            }
        }
    }

    let mut result = None;
    for start in marker_starts {
        if let Some(probs) = extract_at_marker(logprobs, &tokens, start, min_logprob_coverage) {
            result = Some(probs);
        }
    }
    result
}

/// Scan the 10 tokens following one "Verdict" marker for the verdict letter
/// and build the A-D distribution from its top_logprobs. Returns None if no
/// letter token is found or the A-D mass is below the coverage threshold.
fn extract_at_marker(
    logprobs: &[LogprobContent],
    tokens: &[&str],
    search_start: usize,
    min_logprob_coverage: f64,
) -> Option<[f64; 4]> {
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
            _ => return None,
        };

        let mut choice_probs = [0.0_f64; 4];

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
            // Normalize into a proper distribution over A-D.
            for p in &mut choice_probs {
                *p /= prob_sum;
            }
            return Some(choice_probs);
        } else {
            // Logprobs don't cover enough of the A-D space — parse fails.
            return None;
        }
    }

    None
}

/// Parse a comparison response into the judge's categorical verdict distribution.
///
/// Logprobs only — no text fallback. Returns None if logprob extraction fails.
pub fn parse_response(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> ParseResult {
    ParseResult {
        category_probs: extract_likert_probabilities(logprobs, min_logprob_coverage),
    }
}

/// Parse a verdict letter from response text (for --no-logprobs mode).
///
/// Finds the last "Verdict [A-D]" in the text and returns a one-hot distribution
/// over the four categories. Uses the last occurrence to handle cases where
/// "verdict" appears in the analysis before the final verdict.
pub fn parse_response_text(text: &str) -> ParseResult {
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
                        // Word-boundary check: the letter must stand alone.
                        // "Verdict: Draw" is not Verdict D, "Verdict: Both"
                        // is not Verdict B.
                        let next_byte = after_verdict + byte_off + c.len_utf8();
                        let followed_by_word = lower[next_byte..]
                            .chars()
                            .next()
                            .is_some_and(|nc| nc.is_alphanumeric());
                        if followed_by_word {
                            break;
                        }
                        result = Some(idx);
                    }
                    break;
                }
            }
        }
        search_start = after_verdict;
    }

    ParseResult {
        category_probs: result.map(one_hot),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letter_to_index() {
        assert_eq!(letter_to_index('A'), Some(0));
        assert_eq!(letter_to_index('a'), Some(0));
        assert_eq!(letter_to_index('D'), Some(3));
        assert_eq!(letter_to_index('E'), None);
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

        let probs = extract_likert_probabilities(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE)
            .expect("distribution should be Some");
        // B dominates, so P(B) (index 1) should carry almost all the mass.
        assert!(probs[1] > 0.9, "P(B) {} should be > 0.9", probs[1]);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9, "distribution should normalize");
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

        let probs = parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("should parse");
        assert!(probs[1] > probs[0], "B should outweigh A");
    }

    #[test]
    fn test_logprob_parse_uses_last_verdict_not_prose() {
        // Law analyses say "verdict" in prose ("reach a fair verdict") before the
        // real final "Verdict B:". The parser must use the LAST marker, not the first.
        let logprobs = vec![
            LogprobContent { token: "reach".to_string(), top_logprobs: None },
            LogprobContent { token: " a".to_string(), top_logprobs: None },
            LogprobContent { token: " fair".to_string(), top_logprobs: None },
            // Prose "verdict" — first occurrence, must be ignored.
            LogprobContent { token: " verdict".to_string(), top_logprobs: None },
            LogprobContent { token: ".".to_string(), top_logprobs: None },
            LogprobContent { token: " Citing".to_string(), top_logprobs: None },
            LogprobContent { token: " the".to_string(), top_logprobs: None },
            // The real final marker.
            LogprobContent { token: "Verdict".to_string(), top_logprobs: None },
            LogprobContent {
                token: " B".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "B".to_string(), logprob: -0.05 },
                    TopLogprob { token: "A".to_string(), logprob: -3.5 },
                    TopLogprob { token: "C".to_string(), logprob: -4.0 },
                    TopLogprob { token: "D".to_string(), logprob: -5.0 },
                    TopLogprob { token: "E".to_string(), logprob: -6.0 },
                ]),
            },
        ];

        let probs = parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("should find the final Verdict B:");
        assert!(probs[1] > probs[0], "B at the final marker should dominate A");
    }

    #[test]
    fn test_min_logprob_coverage_gates_parsing() {
        // A-D mass sums to 0.96 (the rest is on a non-letter token).
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
        assert!(parse_response(&logprobs, 0.95).category_probs.is_some());

        // ...but not a stricter 0.99 threshold.
        assert!(parse_response(&logprobs, 0.99).category_probs.is_none());
    }

    #[test]
    fn test_parse_response_no_logprobs_returns_none() {
        let result = parse_response(&[], DEFAULT_MIN_LOGPROB_COVERAGE);
        assert!(result.category_probs.is_none());
    }

    // --- Text-based parsing tests (--no-logprobs mode) ---

    #[test]
    fn test_text_parse_verdict_b() {
        let text = "Some analysis.\n\nVerdict B: Option 1, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_verdict_a() {
        let text = "Analysis here.\n\nVerdict A: Option 1, clearly";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_text_parse_verdict_d() {
        let text = "Analysis here.\n\nVerdict D: Option 2, clearly";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(3)));
    }

    #[test]
    fn test_text_parse_verdict_c() {
        let text = "Analysis here.\n\nVerdict C: Option 2, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(2)));
    }

    #[test]
    fn test_text_parse_verdict_with_colon_separator() {
        let text = "Analysis.\n\nVerdict: B: Option 1, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_uses_last_verdict() {
        // "verdict" appears in the analysis, but we want the final one
        let text = "The verdict on flavor is mixed.\n\nVerdict D: Option 2, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(3)));
    }

    #[test]
    fn test_text_parse_no_verdict_returns_none() {
        let text = "Some analysis without a final answer.";
        assert!(parse_response_text(text).category_probs.is_none());
    }

    #[test]
    fn test_text_parse_case_insensitive() {
        let text = "Analysis.\n\nVERDICT B: Option 1, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_lowercase_verdict_letter() {
        let text = "Analysis.\n\nVerdict b: Option 1, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_bold_wrapped_verdict() {
        let text = "Analysis.\n\n**Verdict D**";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(3)));
    }

    #[test]
    fn test_text_parse_bold_verdict_with_colon() {
        let text = "Analysis.\n\n**Verdict B:** Option 1, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_bold_only_keyword() {
        let text = "Analysis.\n\n**Verdict** A";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_text_parse_heading_verdict() {
        let text = "Analysis.\n\n## Verdict D: Option 2, marginally";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(3)));
    }

    #[test]
    fn test_text_parse_rejects_word_starting_with_letter() {
        // A verdict line starting with a WORD that begins with A-D must not
        // parse as that letter.
        assert!(parse_response_text("Analysis.\n\nVerdict: Draw").category_probs.is_none());
        assert!(parse_response_text("Analysis.\n\nVerdict: Both are good").category_probs.is_none());
        assert!(parse_response_text("Analysis.\n\nVerdict: apple wins").category_probs.is_none());
    }

    #[test]
    fn test_text_parse_letter_followed_by_punctuation_still_parses() {
        assert_eq!(parse_response_text("Analysis.\n\nVerdict: B.").category_probs, Some(one_hot(1)));
        assert_eq!(parse_response_text("Analysis.\n\nVerdict: A, clearly").category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_logprob_parse_prose_verdict_after_real_verdict() {
        // The real "Verdict: B" comes first; prose containing "verdict"
        // follows. The parser must keep the successful extraction instead of
        // latching onto the later prose marker and discarding everything.
        let logprobs = vec![
            LogprobContent { token: "Verdict".to_string(), top_logprobs: None },
            LogprobContent { token: ":".to_string(), top_logprobs: None },
            LogprobContent {
                token: " B".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "B".to_string(), logprob: -0.05 },
                    TopLogprob { token: "A".to_string(), logprob: -3.5 },
                    TopLogprob { token: "C".to_string(), logprob: -4.0 },
                    TopLogprob { token: "D".to_string(), logprob: -5.0 },
                    TopLogprob { token: "E".to_string(), logprob: -6.0 },
                ]),
            },
            LogprobContent { token: "\n".to_string(), top_logprobs: None },
            LogprobContent { token: "This".to_string(), top_logprobs: None },
            LogprobContent { token: " verdict".to_string(), top_logprobs: None },
            LogprobContent { token: " reflects".to_string(), top_logprobs: None },
            LogprobContent { token: " the".to_string(), top_logprobs: None },
            LogprobContent { token: " analysis".to_string(), top_logprobs: None },
            LogprobContent { token: ".".to_string(), top_logprobs: None },
        ];

        let probs = parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("the real Verdict: B must survive trailing prose mentioning 'verdict'");
        assert!(probs[1] > 0.9, "P(B) {} should dominate", probs[1]);
    }

    #[test]
    fn test_logprob_parse_no_verdict_marker_discards() {
        // The judge is instructed to write "Verdict: x" — a bare letter with
        // no marker anywhere must be discarded, not scanned for.
        let logprobs = vec![
            LogprobContent { token: "B".to_string(), top_logprobs: Some(vec![
                TopLogprob { token: "B".to_string(), logprob: -0.05 },
                TopLogprob { token: "A".to_string(), logprob: -3.5 },
            ]) },
            LogprobContent { token: " is".to_string(), top_logprobs: None },
            LogprobContent { token: " better".to_string(), top_logprobs: None },
        ];
        assert!(parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs.is_none());
    }
}
