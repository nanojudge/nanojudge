/// Logprob extraction for likert verdicts (logprobs only, no text fallback).
use serde::Deserialize;

/// Default narrow-win probability (B and D on the likert scale).
pub const DEFAULT_NARROW_WIN: f64 = 0.8;

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
    /// P(item1 wins), from logprobs. None if logprob extraction failed.
    pub item1_win_probability: Option<f64>,
}

fn letter_to_index(c: char) -> Option<usize> {
    LIKERT_LETTERS.iter().position(|&l| l == c.to_ascii_uppercase())
}

/// Extract likert choice probabilities from logprobs.
///
/// Returns (choice_probs, expected_p1) or (None, None) if extraction fails.
fn extract_likert_probabilities(logprobs: &[LogprobContent], mapping: &[f64; 5]) -> (Option<[f64; 5]>, Option<f64>) {
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
        if prob_sum >= 0.99 {
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

/// Fallback: extract rating letter from the last few lines of response text.
///
/// Three-pass system (see inline comments for each pass).
/// Returns the likert index (0-4) if found.
fn parse_rating_from_text(text: &str) -> Option<usize> {
    let lines: Vec<&str> = text.trim().lines().collect();
    let start = lines.len().saturating_sub(10);

    // Pass 1: scan last lines for explicit patterns (reverse order)
    for line in lines[start..].iter().rev() {
        // Strip markdown bold markers before matching
        let trimmed = line.trim().replace("**", "");
        let trimmed = trimmed.trim();

        // Try "X: ..." at start of line
        let chars: Vec<char> = trimmed.chars().take(2).collect();
        if chars.len() >= 2 && chars[1] == ':' {
            if let Some(idx) = letter_to_index(chars[0]) {
                return Some(idx);
            }
        }

        // Try "Verdict: X" / "Verdict X:" or "Verdict: Option X" patterns on the same line
        let lower = trimmed.to_lowercase();
        // Find the keyword and its length
        let marker = lower.find("verdict").map(|pos| (pos, 7));
        if let Some((pos, keyword_len)) = marker {
            let after_keyword = &lower[pos + keyword_len..];
            // Skip colon, spaces, etc. after the keyword
            let cleaned = after_keyword
                .trim_start_matches(|c: char| c == ':' || c == ' ' || c == '\t');
            // Strip "option" prefix if present (e.g. "Option A clearly wins")
            let cleaned = cleaned.strip_prefix("option").unwrap_or(cleaned)
                .trim_start();
            if let Some(ch) = cleaned.chars().next() {
                if let Some(idx) = letter_to_index(ch) {
                    return Some(idx);
                }
            }
        }
    }

    // Pass 2: cross-line pattern — "Verdict:" on one line, bare letter on the next
    for i in start..lines.len().saturating_sub(1) {
        let cur = lines[i].trim().replace("**", "");
        let cur_lower = cur.trim().to_lowercase();
        if cur_lower.ends_with("verdict:") {
            // Next non-empty line should be a bare letter
            for j in (i + 1)..lines.len() {
                let next = lines[j].trim();
                if next.is_empty() { continue; }
                let first = next.chars().next().unwrap();
                if let Some(idx) = letter_to_index(first) {
                    return Some(idx);
                }
                break; // first non-empty line after keyword wasn't A-E
            }
        }
    }

    // Pass 3: phrase-match fallback — scan entire text for known verdict phrases.
    // Searches for the last occurrence of any verdict sentence in the full text.
    // This catches cases where the model writes "Option 1 clearly wins" without "Verdict",
    // or where "Verdict X:" appears inline and earlier passes missed it.
    let verdict_phrases: &[(&str, usize)] = &[
        ("option 1 clearly wins", 0),
        ("option 1 narrowly wins", 1),
        ("option 2 narrowly wins", 3),
        ("option 2 clearly wins", 4),
    ];
    let full_lower = text.to_lowercase();
    let mut best: Option<(usize, usize)> = None; // (position, likert_index)
    for &(phrase, idx) in verdict_phrases {
        if let Some(pos) = full_lower.rfind(phrase) {
            match best {
                None => best = Some((pos, idx)),
                Some((prev_pos, _)) if pos > prev_pos => best = Some((pos, idx)),
                _ => {}
            }
        }
    }
    if let Some((_, idx)) = best {
        return Some(idx);
    }

    None
}

/// Parse a comparison response into a win probability.
///
/// Logprobs only — no text fallback. Returns None if logprob extraction fails.
pub fn parse_response(_text: &str, logprobs: &[LogprobContent], narrow_win: f64) -> ParseResult {
    let mapping = likert_mapping(narrow_win);
    let (_, expected_p1) = extract_likert_probabilities(logprobs, &mapping);

    ParseResult {
        item1_win_probability: expected_p1,
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
    fn test_parse_rating_from_text() {
        let text = "Some analysis here.\n\nVerdict:\nA: Option 1 clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(0));

        let text = "Analysis.\nD: Option 2 narrowly wins";
        assert_eq!(parse_rating_from_text(text), Some(3));

        let text = "No verdict here at all.";
        assert_eq!(parse_rating_from_text(text), None);

        // "Verdict: X:" on the same line
        let text = "Some analysis.\n\nVerdict: A: Option 1 clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(0));

        let text = "Some analysis.\n\nVerdict: E: Option 2 clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(4));

        // "verdict:" lowercase
        let text = "Analysis paragraph.\n\nverdict: B: Option 1 narrowly wins";
        assert_eq!(parse_rating_from_text(text), Some(1));

        // "Verdict: Option X" format (model writes "Option" before the letter)
        let text = "Long paragraph here. Verdict: Option A clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(0));

        let text = "Analysis.\nVerdict: Option E clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(4));

        // Markdown bold **Verdict:**
        let text = "Analysis here.\n\n**Verdict:** A";
        assert_eq!(parse_rating_from_text(text), Some(0));

        // "Verdict X:" atomic format (new prompt style)
        let text = "Analysis.\n\nVerdict A: Option 1 clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(0));

        let text = "Analysis.\n\nVerdict E: Option 2 clearly wins";
        assert_eq!(parse_rating_from_text(text), Some(4));

        let text = "Analysis.\n\nVerdict D: Option 2 narrowly wins";
        assert_eq!(parse_rating_from_text(text), Some(3));

        // Cross-line: "Verdict:" on one line, bare letter on next
        let text = "Analysis.\n\nVerdict:\nE";
        assert_eq!(parse_rating_from_text(text), Some(4));

        let text = "Analysis.\n\nVerdict:\n\nA";
        assert_eq!(parse_rating_from_text(text), Some(0));

        // Cross-line with bold
        let text = "Analysis.\n\n**Verdict:**\nD";
        assert_eq!(parse_rating_from_text(text), Some(3));

        // Stop-sequence truncated: "Verdict X:" at end of paragraph (no label after)
        let text = "Long paragraph about things. Verdict D:";
        assert_eq!(parse_rating_from_text(text), Some(3));

        let text = "Analysis here.\n\nTherefore Option 1 wins. Verdict A:";
        assert_eq!(parse_rating_from_text(text), Some(0));

        // Phrase-match fallback: "Option X clearly/narrowly wins" without Verdict prefix
        let text = "Analysis.\n\nTherefore, Option 1 clearly wins.";
        assert_eq!(parse_rating_from_text(text), Some(0));

        let text = "Analysis.\n\nTherefore, Option 2 clearly wins.";
        assert_eq!(parse_rating_from_text(text), Some(4));

        let text = "Therefore, Option 2 narrowly wins.";
        assert_eq!(parse_rating_from_text(text), Some(3));

        let text = "Therefore, Option 1 narrowly wins.";
        assert_eq!(parse_rating_from_text(text), Some(1));
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
        let (choice_probs, expected_p1) = extract_likert_probabilities(&logprobs, &mapping);
        assert!(choice_probs.is_some(), "choice_probs should be Some");
        assert!(expected_p1.is_some(), "expected_p1 should be Some");

        let p1 = expected_p1.unwrap();
        // B is dominant (0.8 mapping), so expected_p1 should be close to 0.8
        assert!(p1 > 0.7, "expected_p1 {p1} should be > 0.7");
        assert!(p1 < 0.9, "expected_p1 {p1} should be < 0.9");
    }

    #[test]
    fn test_parse_response_with_logprobs() {
        let text = "Analysis text.\n\nVerdict:\nB: Option 1 narrowly wins";
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

        let result = parse_response(text, &logprobs, DEFAULT_NARROW_WIN);
        assert!(result.item1_win_probability.is_some());
        let p = result.item1_win_probability.unwrap();
        assert!(p > 0.7);
    }

    #[test]
    fn test_parse_response_no_logprobs_returns_none() {
        // Without logprobs, parse_response returns None (no text fallback)
        let text = "Some analysis.\n\nVerdict:\nD: Option 2 narrowly wins";
        let result = parse_response(text, &[], DEFAULT_NARROW_WIN);
        assert!(result.item1_win_probability.is_none());
    }

    #[test]
    fn test_parse_response_unparseable() {
        let text = "I don't know what to say.";
        let result = parse_response(text, &[], DEFAULT_NARROW_WIN);
        assert!(result.item1_win_probability.is_none());
    }

    #[test]
    fn test_custom_narrow_win_no_logprobs_returns_none() {
        // Without logprobs, parse_response returns None regardless of text or narrow_win value
        let text = "Analysis.\n\nVerdict:\nB: Option 1 narrowly wins";
        let result = parse_response(text, &[], 0.7);
        assert!(result.item1_win_probability.is_none());

        let text = "Analysis.\n\nVerdict:\nD: Option 2 narrowly wins";
        let result = parse_response(text, &[], 0.7);
        assert!(result.item1_win_probability.is_none());
    }
}
