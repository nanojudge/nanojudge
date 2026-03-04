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
