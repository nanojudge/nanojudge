/// Verdict extraction for LLM judgements.
///
/// The verdict line is "Verdict: Option 1" or "Verdict: Option 2". Two
/// separate parsing modes:
/// - Logprob mode: extracts continuous probabilities from the option-digit
///   token's logprobs.
/// - Text mode (--no-logprobs): extracts the discrete verdict from response text.
use nanojudge_core::LineupVerdict;
use nanojudge_core::constants::MAX_LINEUP_SIZE;
use serde::Deserialize;

/// Default minimum fraction of option-digit probability mass the top-logprobs
/// must cover before the logprob-derived distribution is trusted. Below this,
/// the judgement parse fails (returns None).
pub const DEFAULT_MIN_LOGPROB_COVERAGE: f64 = 0.95;

/// One-hot verdict distribution for a single chosen option index.
fn one_hot(idx: usize) -> [f64; 2] {
    let mut v = [0.0; 2];
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

/// Result of parsing a judgement response.
pub struct ParseResult {
    /// Verdict distribution `[P(Option 1 wins), P(Option 2 wins)]` — item1 is
    /// shown as Option 1. None if extraction failed.
    pub category_probs: Option<[f64; 2]>,
}

/// Map an option digit to its verdict index (1 → 0, 2 → 1).
fn digit_to_index(c: char) -> Option<usize> {
    match c {
        '1' => Some(0),
        '2' => Some(1),
        _ => None,
    }
}

/// Concatenate all token strings and record each token's exclusive end byte
/// offset, so any byte position in the combined text maps back to the token
/// that contains it.
fn tokens_with_ends(logprobs: &[LogprobContent]) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut token_ends = Vec::with_capacity(logprobs.len());
    for lp in logprobs {
        text.push_str(&lp.token);
        token_ends.push(text.len());
    }
    (text, token_ends)
}

/// Find every "Verdict: Option <digit>" match in `text`, returning
/// `(digit_byte_pos, option_idx)` in order. Shared by both text mode and
/// logprobs mode so the two parsers accept the exact same grammar.
fn find_pairwise_verdicts(text: &str) -> Vec<(usize, usize)> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut matches = Vec::new();
    let mut search_start = 0;

    while let Some(offset) = lower[search_start..].find("verdict") {
        let after_verdict = search_start + offset + 7; // len("verdict")

        // Skip separators and markdown decoration between "Verdict" and "Option".
        let mut pos = after_verdict;
        while pos < bytes.len() && matches!(bytes[pos], b' ' | b'\t' | b'\n' | b'\r' | b':' | b'*' | b'#') {
            pos += 1;
        }

        if lower[pos..].starts_with("option") {
            // Skip separators between "Option" and the digit.
            let mut dpos = pos + 6; // len("option")
            while dpos < bytes.len() && matches!(bytes[dpos], b' ' | b'\t' | b':' | b'*') {
                dpos += 1;
            }
            if let Some(c) = lower[dpos..].chars().next()
                && let Some(idx) = digit_to_index(c)
            {
                // The digit must stand alone: "Option 12" is not a verdict.
                let followed_by_word = lower[dpos + c.len_utf8()..]
                    .chars()
                    .next()
                    .is_some_and(|nc| nc.is_alphanumeric());
                if !followed_by_word {
                    matches.push((dpos, idx));
                }
            }
        }
        search_start = after_verdict;
    }

    matches
}

/// Extract the judge's verdict distribution `[P(Option 1), P(Option 2)]` from
/// logprobs. Uses the exact same grammar as `parse_response_text` and reads
/// only the LAST match; if that final verdict token is not standalone or falls
/// below `min_logprob_coverage`, the whole judgement fails (no falling back to
/// an earlier match).
fn extract_pairwise_probabilities(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> Option<[f64; 2]> {
    if logprobs.is_empty() {
        return None;
    }

    let (text, token_ends) = tokens_with_ends(logprobs);
    let (digit_byte_pos, expected_idx) = *find_pairwise_verdicts(&text).last()?;
    let tok_idx = token_ends.partition_point(|&e| e <= digit_byte_pos);

    let tok = logprobs[tok_idx].token.trim();
    let first_char = tok.chars().next()?;
    if digit_to_index(first_char) != Some(expected_idx) {
        return None;
    }
    // The digit must stand alone (allow a trailing colon/period), so a fused
    // token like "Option1" is never misread.
    let rest = &tok[first_char.len_utf8()..];
    if !rest.is_empty() && rest != ":" && rest != "." {
        return None;
    }

    let top_logprobs = match &logprobs[tok_idx].top_logprobs {
        Some(tlps) if !tlps.is_empty() => tlps,
        _ => return None,
    };

    let mut choice_probs = [0.0_f64; 2];
    for tlp in top_logprobs {
        let clean = tlp.token.trim().trim_end_matches([':', '.']);
        if clean.len() == 1
            && let Some(tidx) = digit_to_index(clean.chars().next().unwrap())
        {
            choice_probs[tidx] += tlp.logprob.exp();
        }
    }

    let prob_sum: f64 = choice_probs.iter().sum();
    if prob_sum >= min_logprob_coverage && prob_sum > 0.0 {
        for p in &mut choice_probs {
            *p /= prob_sum;
        }
        Some(choice_probs)
    } else {
        None
    }
}

/// Parse a judgement response into the judge's verdict distribution.
///
/// Logprobs only — no text fallback. Returns None if logprob extraction fails.
pub fn parse_response(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> ParseResult {
    ParseResult {
        category_probs: extract_pairwise_probabilities(logprobs, min_logprob_coverage),
    }
}

// ---------------------------------------------------------------------------
// Lineup parsing
// ---------------------------------------------------------------------------
//
// A lineup judgement asks the judge to rank `k` options (2 ≤ k ≤ 9) labelled
// A..I, one "Nth place is Option X" line per rank. Reading the logprobs of the
// first `k - 1` of those lines gives each place's probability for its pick
// among the options still unplaced — see `parse_lineup`.

/// The option letters of a lineup judgement, in order (A..I).
const LINEUP_LETTERS: [char; MAX_LINEUP_SIZE] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];

/// The letter index for `c`, if it labels one of a `lineup_size` lineup's options.
/// Letters past the lineup's size are not options (in a 3-item lineup, "D" is prose).
/// Only uppercase letters label options: lowercase "a" is prose, never Option A.
fn lineup_letter_to_index(c: char, lineup_size: usize) -> Option<usize> {
    LINEUP_LETTERS[..lineup_size].iter().position(|&l| l == c)
}

/// Find every standalone uppercase "\boption\b <seps> <A..I>" mention in `text`,
/// returning `(letter_byte_pos, option_idx)` in order. Shared by both text mode
/// and logprobs mode so the two parsers accept the exact same grammar.
fn find_lineup_options(text: &str, lineup_size: usize) -> Vec<(usize, usize)> {
    let lower = text.to_ascii_lowercase();
    let bytes = text.as_bytes();
    let mut matches = Vec::new();
    let mut search = 0;

    while let Some(rel) = lower[search..].find("option") {
        let begin = search + rel;
        let end = begin + "option".len();
        search = end;

        // Word boundaries on both sides: reject "description A", "options A",
        // "optional A", etc.
        let letter_before = begin > 0 && bytes[begin - 1].is_ascii_alphabetic();
        let letter_after = end < bytes.len() && bytes[end].is_ascii_alphabetic();
        if letter_before || letter_after {
            continue;
        }

        // Skip inline separators and bold markers (no newlines, so prose like
        // "the best option\nA second point..." is never read as Option A).
        let mut pos = end;
        while pos < bytes.len() && matches!(bytes[pos], b' ' | b'\t' | b':' | b'*') {
            pos += 1;
        }

        if let Some(c) = text[pos..].chars().next()
            && let Some(idx) = lineup_letter_to_index(c, lineup_size)
        {
            // Must stand alone: next char is not alphanumeric (so "Option Also" is not A).
            let next = text[pos + c.len_utf8()..].chars().next();
            if next.is_none_or(|nc| !nc.is_alphanumeric()) {
                matches.push((pos, idx));
            }
        }
    }

    matches
}

/// Read a lineup response's logprobs into a [`LineupVerdict`]: the emitted
/// ranking, plus each place's probability for its pick.
///
/// Uses the exact same grammar as `parse_lineup_text`: take the LAST
/// `lineup_size` standalone uppercase "Option <letter>" mentions (the trailing
/// ranking block) and require them to be a full permutation of `0..lineup_size`.
///
/// Each ranking line's logprobs are the judge's top-1 distribution *conditional
/// on the items ranked above it already being placed*. For every place but the
/// last, the mass on options still unplaced must clear `min_logprob_coverage`
/// (mass spent on an already-placed duplicate option does not count), and the
/// place's probability is its emitted letter's share of that unplaced mass.
///
/// The last place is exempt from coverage: one option is left by then, so its
/// letter is fixed by the others and its distribution is never read. Its line
/// and clean letter token must still be present.
pub fn parse_lineup(
    logprobs: &[LogprobContent],
    lineup_size: usize,
    min_logprob_coverage: f64,
) -> Option<LineupVerdict> {
    if logprobs.is_empty() {
        return None;
    }

    let (text, token_ends) = tokens_with_ends(logprobs);
    let matches = find_lineup_options(&text, lineup_size);
    if matches.len() < lineup_size {
        return None;
    }
    let block = &matches[matches.len() - lineup_size..];

    let mut seen = vec![false; lineup_size];
    for &(_, letter) in block {
        seen[letter] = true;
    }
    if !seen.iter().all(|&b| b) {
        return None;
    }

    let mut placed = vec![false; lineup_size];
    let mut place_probs = Vec::with_capacity(lineup_size - 1);

    for (rank, &(byte_pos, letter)) in block.iter().enumerate() {
        let tok_idx = token_ends.partition_point(|&e| e <= byte_pos);
        let tok = logprobs[tok_idx].token.trim();
        let first_char = tok.chars().next()?;
        if lineup_letter_to_index(first_char, lineup_size) != Some(letter) {
            return None;
        }
        let rest = &tok[first_char.len_utf8()..];
        if !rest.is_empty() && rest != ":" && rest != "." {
            return None;
        }

        if rank == lineup_size - 1 {
            continue;
        }

        let mut raw_dist = vec![0.0_f64; lineup_size];
        for tlp in logprobs[tok_idx].top_logprobs.iter().flatten() {
            let clean = tlp.token.trim().trim_end_matches([':', '.']);
            if clean.len() == 1
                && let Some(tidx) =
                    lineup_letter_to_index(clean.chars().next().unwrap(), lineup_size)
            {
                raw_dist[tidx] += tlp.logprob.exp();
            }
        }

        let unplaced_mass: f64 = (0..lineup_size)
            .filter(|&i| !placed[i])
            .map(|i| raw_dist[i])
            .sum();
        if unplaced_mass < min_logprob_coverage || unplaced_mass <= 0.0 {
            return None;
        }
        place_probs.push(raw_dist[letter] / unplaced_mass);
        placed[letter] = true;
    }

    Some(LineupVerdict {
        ranking: block.iter().map(|&(_, letter)| letter).collect(),
        place_probs,
    })
}

/// Parse a lineup ranking from response text (--no-logprobs mode).
///
/// Same rule as the logprob path: take the LAST `lineup_size` "Option <letter>"
/// mentions (the trailing ranking block) and require them to be a distinct full
/// ranking. Without logprobs every pick is hard, so each place's probability is
/// 1.0 — the full ordering is kept, not just the winner. Returns None if a clean
/// full ranking can't be read (throw it out).
pub fn parse_lineup_text(text: &str, lineup_size: usize) -> Option<LineupVerdict> {
    let matches = find_lineup_options(text, lineup_size);
    if matches.len() < lineup_size {
        return None;
    }
    let block = &matches[matches.len() - lineup_size..];
    let mut seen = vec![false; lineup_size];
    for &(_, l) in block {
        seen[l] = true;
    }
    if !seen.iter().all(|&b| b) {
        return None;
    }

    Some(LineupVerdict {
        ranking: block.iter().map(|&(_, l)| l).collect(),
        place_probs: vec![1.0; lineup_size - 1],
    })
}

/// Parse a verdict from response text (for --no-logprobs mode).
///
/// Finds the last "Verdict: Option <digit>" in the text and returns a one-hot
/// distribution over the two options. Uses the last occurrence to handle cases
/// where "verdict" appears in the analysis before the final verdict.
pub fn parse_response_text(text: &str) -> ParseResult {
    ParseResult {
        category_probs: find_pairwise_verdicts(text).last().map(|&(_, idx)| one_hot(idx)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanojudge_core::constants::MIN_LINEUP_SIZE;

    #[test]
    fn test_digit_to_index() {
        assert_eq!(digit_to_index('1'), Some(0));
        assert_eq!(digit_to_index('2'), Some(1));
        assert_eq!(digit_to_index('0'), None);
        assert_eq!(digit_to_index('3'), None);
        assert_eq!(digit_to_index('A'), None);
    }

    #[test]
    fn test_extract_pairwise_from_logprobs() {
        // Simulate logprobs where the model outputs "Verdict: Option" then "1"
        // with top_logprobs showing strong preference for option 1.
        let logprobs = vec![
            plain("Verdict"),
            plain(":"),
            plain(" Option"),
            LogprobContent {
                token: " 1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: -0.05 },   // ~0.951
                    TopLogprob { token: "2".to_string(), logprob: -3.5 },    // ~0.030
                    TopLogprob { token: "3".to_string(), logprob: -6.0 },    // ~0.002, not a verdict digit
                ]),
            },
        ];

        let probs = extract_pairwise_probabilities(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE)
            .expect("distribution should be Some");
        // Option 1 dominates, so P(Option 1) should carry almost all the mass.
        assert!(probs[0] > 0.9, "P(Option 1) {} should be > 0.9", probs[0]);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9, "distribution should normalize");
    }

    #[test]
    fn test_parse_response_with_logprobs() {
        let logprobs = vec![
            plain("Verdict"),
            plain(":"),
            plain("\n"),
            plain("Option"),
            LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "2".to_string(), logprob: -0.05 },
                    TopLogprob { token: "1".to_string(), logprob: -3.5 },
                ]),
            },
        ];

        let probs = parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("should parse");
        assert!(probs[1] > probs[0], "Option 2 should outweigh Option 1");
    }

    #[test]
    fn test_logprob_parse_uses_last_verdict_not_prose() {
        // Law analyses say "verdict" in prose ("reach a fair verdict") before the
        // real final "Verdict: Option 2". The parser must use the LAST marker,
        // not the first.
        let logprobs = vec![
            plain("reach"), plain(" a"), plain(" fair"),
            // Prose "verdict" — first occurrence, must be ignored.
            plain(" verdict"), plain("."),
            plain(" Citing"), plain(" the"),
            // The real final marker.
            plain("Verdict"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "2".to_string(), logprob: -0.05 },
                    TopLogprob { token: "1".to_string(), logprob: -3.5 },
                ]),
            },
        ];

        let probs = parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("should find the final Verdict: Option 2");
        assert!(probs[1] > probs[0], "Option 2 at the final marker should dominate");
    }

    #[test]
    fn test_min_logprob_coverage_gates_parsing() {
        // Digit mass sums to 0.96 (the rest is on a non-digit token).
        let logprobs = vec![
            plain("Verdict"), plain(":"), plain(" Option"),
            LogprobContent {
                token: "1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: (0.96_f64).ln() },
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
    fn test_text_parse_option_1() {
        let text = "Some analysis.\n\nVerdict: Option 1";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_text_parse_option_2() {
        let text = "Analysis here.\n\nVerdict: Option 2";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_verdict_without_colon() {
        let text = "Analysis here.\n\nVerdict Option 2";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_uses_last_verdict() {
        // "verdict" appears in the analysis, but we want the final one
        let text = "The verdict on flavor is mixed.\n\nVerdict: Option 2";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_no_verdict_returns_none() {
        let text = "Some analysis without a final answer.";
        assert!(parse_response_text(text).category_probs.is_none());
    }

    #[test]
    fn test_text_parse_case_insensitive() {
        let text = "Analysis.\n\nVERDICT: OPTION 1";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_text_parse_bold_wrapped_verdict() {
        let text = "Analysis.\n\n**Verdict: Option 2**";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_bold_only_keyword() {
        let text = "Analysis.\n\n**Verdict**: Option 1";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_text_parse_bold_option() {
        let text = "Analysis.\n\nVerdict: **Option 1**";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_text_parse_heading_verdict() {
        let text = "Analysis.\n\n## Verdict: Option 2";
        assert_eq!(parse_response_text(text).category_probs, Some(one_hot(1)));
    }

    #[test]
    fn test_text_parse_rejects_missing_option_word() {
        // The instructed format is "Verdict: Option <n>" — a bare digit is not
        // a verdict.
        assert!(parse_response_text("Analysis.\n\nVerdict: 1").category_probs.is_none());
    }

    #[test]
    fn test_text_parse_rejects_other_digits() {
        assert!(parse_response_text("Analysis.\n\nVerdict: Option 3").category_probs.is_none());
        // Multi-digit numbers are not verdicts.
        assert!(parse_response_text("Analysis.\n\nVerdict: Option 12").category_probs.is_none());
    }

    #[test]
    fn test_text_parse_rejects_prose_after_verdict() {
        assert!(parse_response_text("Analysis.\n\nVerdict: Both are good").category_probs.is_none());
        assert!(parse_response_text("Analysis.\n\nVerdict: optional 1 extra").category_probs.is_none());
    }

    #[test]
    fn test_text_parse_digit_followed_by_punctuation_still_parses() {
        assert_eq!(parse_response_text("Analysis.\n\nVerdict: Option 2.").category_probs, Some(one_hot(1)));
        assert_eq!(parse_response_text("Analysis.\n\nVerdict: Option 1, clearly").category_probs, Some(one_hot(0)));
    }

    #[test]
    fn test_logprob_parse_prose_verdict_after_real_verdict() {
        // The real "Verdict: Option 2" comes first; prose containing "verdict"
        // follows. The parser must keep the successful extraction instead of
        // latching onto the later prose marker and discarding everything.
        let logprobs = vec![
            plain("Verdict"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "2".to_string(), logprob: -0.05 },
                    TopLogprob { token: "1".to_string(), logprob: -3.5 },
                ]),
            },
            plain("\n"), plain("This"), plain(" verdict"), plain(" reflects"),
            plain(" the"), plain(" analysis"), plain("."),
        ];

        let probs = parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("the real Verdict: Option 2 must survive trailing prose mentioning 'verdict'");
        assert!(probs[1] > 0.9, "P(Option 2) {} should dominate", probs[1]);
    }

    #[test]
    fn test_logprob_parse_deepseek_three_token_marker_split() {
        // DeepSeek-style tokenization: the marker arrives split as
        // "V"+"erd"+"ict" — three tokens — and the digit is a standalone " 2"
        // token whose top_logprobs hold the verdict mass.
        let lp = vec![
            plain(" a"), plain(" cat"), plain(".\n\n"),
            plain("V"), plain("erd"), plain("ict"),
            plain(":"), plain(" Option"),
            LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: " 2".to_string(), logprob: -0.002 },
                    TopLogprob { token: " 1".to_string(), logprob: -6.488 },
                    TopLogprob { token: ":".to_string(), logprob: -17.279 },
                ]),
            },
        ];
        let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("DeepSeek V+erd+ict marker split must parse");
        assert!(probs[1] > 0.99, "P(Option 2) {} should dominate", probs[1]);
    }

    #[test]
    fn test_logprob_parse_two_token_marker_splits() {
        // Every two-token split of "Verdict" must be found by the offset-based
        // marker search.
        for (a, b) in [("V", "erdict"), ("Ve", "rdict"), ("Ver", "dict"), ("Verd", "ict"), ("Verdi", "ct"), ("Verdic", "t")] {
            let lp = vec![
                plain("Analysis"), plain(".\n\n"),
                plain(a), plain(b),
                plain(":"), plain(" Option"),
                LogprobContent {
                    token: " 1".to_string(),
                    top_logprobs: Some(vec![
                        TopLogprob { token: "1".to_string(), logprob: -0.05 },
                        TopLogprob { token: "2".to_string(), logprob: -3.5 },
                    ]),
                },
            ];
            let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
                .unwrap_or_else(|| panic!("split {a}+{b} must parse"));
            assert!(probs[0] > 0.9, "split {a}+{b}: P(Option 1) {} should dominate", probs[0]);
        }
    }

    #[test]
    fn test_logprob_parse_split_option_anchor() {
        // The "Option" anchor itself can be split by the tokenizer.
        let lp = vec![
            plain("Verdict"), plain(":"), plain(" Opt"), plain("ion"),
            LogprobContent {
                token: " 1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: -0.05 },
                    TopLogprob { token: "2".to_string(), logprob: -3.5 },
                ]),
            },
        ];
        let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("split Option anchor must parse");
        assert!(probs[0] > 0.9, "P(Option 1) {} should dominate", probs[0]);
    }

    #[test]
    fn test_logprob_parse_verdict_without_option_anchor_discards() {
        // "Verdict: 1" (missing the "Option" word) is not the instructed
        // format — discard rather than guess.
        let logprobs = vec![
            plain("Verdict"), plain(":"),
            LogprobContent {
                token: " 1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: -0.05 },
                    TopLogprob { token: "2".to_string(), logprob: -3.5 },
                ]),
            },
        ];
        assert!(parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs.is_none());
    }

    #[test]
    fn test_logprob_parse_any_tokenization_of_marker() {
        // Simulate arbitrary tokenizers: chunk the response text into
        // pseudo-random 1-4 byte tokens and require the parse to succeed for
        // every chunking. The option digit itself stays a standalone token —
        // that is a hard requirement of logprobs mode (the digit token's
        // top_logprobs ARE the data), not something the parser can recover from.
        let prefix = "The analysis weighs both options and reaches a verdict on merit.\n\nVerdict: Option";
        for seed in 0..200u64 {
            let mut lp = chunk_plain(prefix, seed);
            lp.push(LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: " 2".to_string(), logprob: -0.01 },
                    TopLogprob { token: " 1".to_string(), logprob: -5.0 },
                ]),
            });
            let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
                .unwrap_or_else(|| panic!("seed {seed}: chunked tokenization must parse"));
            assert!(probs[1] > 0.9, "seed {seed}: P(Option 2) {} should dominate", probs[1]);
        }
    }

    // --- Three-item lineup parsing tests ---

    /// Build a logprob token with no top_logprobs (a structural/marker token).
    fn plain(tok: &str) -> LogprobContent {
        LogprobContent { token: tok.to_string(), top_logprobs: None }
    }

    /// Split ASCII `text` into deterministic pseudo-random 1-4 byte chunks
    /// (LCG-seeded), simulating an arbitrary tokenizer. Plain tokens only.
    fn chunk_plain(text: &str, seed: u64) -> Vec<LogprobContent> {
        assert!(text.is_ascii(), "chunker splits at byte granularity");
        let mut state = seed;
        let bytes = text.as_bytes();
        let mut toks = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let len = 1 + ((state >> 33) as usize) % 4;
            let end = (i + len).min(bytes.len());
            toks.push(plain(std::str::from_utf8(&bytes[i..end]).unwrap()));
            i = end;
        }
        toks
    }

    /// Build a ranking letter token whose top_logprobs put the given
    /// probabilities on A, B, C respectively.
    fn letter_tok(tok: &str, pa: f64, pb: f64, pc: f64) -> LogprobContent {
        let mut tlps = Vec::new();
        for (l, p) in [("A", pa), ("B", pb), ("C", pc)] {
            if p > 0.0 {
                tlps.push(TopLogprob { token: l.to_string(), logprob: p.ln() });
            }
        }
        LogprobContent { token: tok.to_string(), top_logprobs: Some(tlps) }
    }

    /// A full three-item lineup ranking block: 1st=A (0.9/0.03/0.07), 2nd=B over the
    /// non-winners (B 0.8, C 0.2).
    fn lineup_logprobs() -> Vec<LogprobContent> {
        vec![
            plain("1st"), plain(":"), plain(" Option"),
            letter_tok(" A", 0.9, 0.03, 0.07),
            plain("\n"),
            plain("2nd"), plain(":"), plain(" Option"),
            letter_tok(" B", 0.0001, 0.8, 0.2),
            plain("\n"),
            plain("3rd"), plain(":"), plain(" Option"),
            letter_tok(" C", 0.0, 0.0, 1.0),
        ]
    }

    /// Assert a lineup verdict's ranking and place probabilities.
    fn assert_verdict(v: &LineupVerdict, ranking: &[usize], place_probs: &[f64]) {
        assert_eq!(v.ranking, ranking, "ranking");
        assert_eq!(v.place_probs.len(), place_probs.len(), "place_probs length");
        for (got, want) in v.place_probs.iter().zip(place_probs) {
            assert!(
                (got - want).abs() < 1e-9,
                "place_probs {:?}, expected {:?}",
                v.place_probs,
                place_probs
            );
        }
    }

    #[test]
    fn test_parse_lineup_reads_place_probabilities() {
        let v = parse_lineup(&lineup_logprobs(), 3, 0.95).expect("should parse");
        // 1st: A at 0.9. 2nd: B's share of the unplaced B/C mass, 0.8 : 0.2
        // (the 0.0001 still on the already-placed A is not counted).
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_certain_first_place_keeps_second_place() {
        // The 1st-place line's top_logprobs hold only A: the pick is certain.
        // The 2nd place's B-over-C probability must still come through.
        let lp = vec![
            plain("1st"), plain(":"), plain(" Option"),
            letter_tok(" A", 1.0, 0.0, 0.0),
            plain("\n"),
            plain("2nd"), plain(":"), plain(" Option"),
            letter_tok(" B", 0.0, 0.8, 0.2),
            plain("\n"),
            plain("3rd"), plain(":"), plain(" Option"),
            letter_tok(" C", 0.0, 0.0, 1.0),
        ];
        let v = parse_lineup(&lp, 3, 0.95).expect("should parse");
        assert_verdict(&v, &[0, 1, 2], &[1.0, 0.8]);
    }

    #[test]
    fn test_parse_lineup_split_ordinal_tokens() {
        // Gemma-style tokenization: "1st" → "1" + "st". The parser must still
        // find the markers and read the letter that follows.
        let lp = vec![
            plain("1"), plain("st"), plain(":"), plain(" Option"),
            letter_tok(" A", 0.9, 0.03, 0.07),
            plain("\n"),
            plain("2"), plain("nd"), plain(":"), plain(" Option"),
            letter_tok(" B", 0.0001, 0.8, 0.2),
            plain("\n"),
            plain("3"), plain("rd"), plain(":"), plain(" Option"),
            letter_tok(" C", 0.0, 0.0, 1.0),
        ];
        let v = parse_lineup(&lp, 3, 0.95).expect("split-ordinal tokens should parse");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_split_option_marker() {
        // DeepSeek-style tokenizer split of the "Option" anchor word itself.
        let lp = vec![
            plain("1st"), plain(":"), plain(" O"), plain("ption"),
            letter_tok(" A", 0.9, 0.03, 0.07),
            plain("\n"),
            plain("2nd"), plain(":"), plain(" Opt"), plain("ion"),
            letter_tok(" B", 0.0001, 0.8, 0.2),
            plain("\n"),
            plain("3rd"), plain(":"), plain(" Op"), plain("t"), plain("ion"),
            letter_tok(" C", 0.0, 0.0, 1.0),
        ];
        let v = parse_lineup(&lp, 3, 0.95).expect("split Option anchors should parse");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_prose_options_word_is_not_an_anchor() {
        // Trailing prose containing "options" followed by a letter-like token
        // must not register as a fourth anchor and shift the ranking block.
        let mut lp = lineup_logprobs();
        lp.extend([plain("\n"), plain("Best"), plain(" of"), plain(" the"), plain(" options")]);
        lp.push(letter_tok(" A", 0.9, 0.05, 0.05));
        let v = parse_lineup(&lp, 3, 0.95).expect("block before the prose must parse");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_any_tokenization_of_markers() {
        // Same arbitrary-chunking guarantee as the pairwise parser: any
        // tokenization of the structural text must parse, with the option
        // letters standalone (hard requirement of logprobs mode).
        for seed in 0..200u64 {
            let mut lp = chunk_plain("The options each have merit.\n\n1st: Option", seed);
            lp.push(letter_tok(" A", 0.9, 0.03, 0.07));
            lp.extend(chunk_plain("\n2nd: Option", seed ^ 1));
            lp.push(letter_tok(" B", 0.0001, 0.8, 0.2));
            lp.extend(chunk_plain("\n3rd: Option", seed ^ 2));
            lp.push(letter_tok(" C", 0.0, 0.0, 1.0));
            let v = parse_lineup(&lp, 3, 0.95)
                .unwrap_or_else(|| panic!("seed {seed}: chunked tokenization must parse"));
            assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
        }
    }

    #[test]
    fn test_parse_lineup_rejects_repeated_option() {
        // Model repeated the same option ("Option B" x3) instead of ranking — not
        // a distinct 1-2-3 ranking, so the whole judgement is thrown out.
        let lp = vec![
            plain(" Option"), letter_tok(" B", 0.02, 0.97, 0.01), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.02, 0.97, 0.01), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.02, 0.97, 0.01),
        ];
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_text_rejects_repeated_option() {
        assert!(parse_lineup_text("analysis\n\nOption B\nOption B\nOption B", 3).is_none());
    }

    #[test]
    fn test_parse_lineup_rejects_fewer_than_three() {
        // Only two clean results → cannot form a full ranking → throw out.
        let lp = vec![
            plain(" Option"), letter_tok(" A", 0.9, 0.05, 0.05), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.05, 0.9, 0.05),
        ];
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_bare_option_lines() {
        // Model dropped the "Nth:" prefixes and ended with bare "Option X" lines.
        // An earlier prose "Option C" mention must be ignored (last three win).
        let lp = vec![
            plain(" Option"), letter_tok(" C", 0.1, 0.2, 0.7), plain(" is"), plain(" best"), plain("."),
            plain("\n"),
            plain(" Option"), letter_tok(" A", 0.9, 0.03, 0.07),
            plain("\n"),
            plain(" Option"), letter_tok(" B", 0.0001, 0.8, 0.2),
            plain("\n"),
            plain(" Option"), letter_tok(" C", 0.0, 0.0, 1.0),
        ];
        let v = parse_lineup(&lp, 3, 0.95).expect("bare Option lines should parse");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_ignores_prose_place_before_block() {
        // "in the first place" style prose (an earlier "first" ordinal) must not
        // derail the block — the LAST "1st"/"first" marker wins.
        let mut lp = vec![plain("in"), plain(" the"), plain(" first"), plain(" place"), plain(".")];
        lp.extend(lineup_logprobs());
        let v = parse_lineup(&lp, 3, 0.95).expect("should still find the final block");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_low_coverage_fails() {
        // 1st-place letter puts only 0.5 mass on A-C (rest on a stray token).
        let lp = vec![
            plain("1st"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " A".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "A".to_string(), logprob: (0.5_f64).ln() },
                    TopLogprob { token: " the".to_string(), logprob: (0.5_f64).ln() },
                ]),
            },
            plain("2nd"), plain(":"), plain(" Option"),
            letter_tok(" B", 0.0, 0.8, 0.2),
        ];
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_last_place_needs_no_coverage() {
        // Constrained decoding with one letter left: most of the last line's mass
        // is on a bare " " token (an alternative spelling of " C"). The ranking
        // is still complete and the last place's distribution is never read.
        let mut lp = lineup_logprobs();
        lp.pop();
        lp.push(LogprobContent {
            token: " C".to_string(),
            top_logprobs: Some(vec![
                TopLogprob { token: " ".to_string(), logprob: (0.89_f64).ln() },
                TopLogprob { token: " C".to_string(), logprob: (0.11_f64).ln() },
            ]),
        });
        let v = parse_lineup(&lp, 3, 0.95).expect("last place is exempt from coverage");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_low_coverage_second_place_fails() {
        let mut lp = lineup_logprobs();
        lp[8] = LogprobContent {
            token: " B".to_string(),
            top_logprobs: Some(vec![
                TopLogprob { token: " ".to_string(), logprob: (0.5_f64).ln() },
                TopLogprob { token: " B".to_string(), logprob: (0.5_f64).ln() },
            ]),
        };
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_missing_second_place_fails() {
        let lp = vec![
            plain("1st"), plain(":"), plain(" Option"),
            letter_tok(" A", 0.9, 0.05, 0.05),
        ];
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_text_keeps_full_ranking() {
        let text = "Analysis here.\n\n1st: Option B\n2nd: Option A\n3rd: Option C";
        assert_eq!(
            parse_lineup_text(text, 3),
            Some(LineupVerdict { ranking: vec![1, 0, 2], place_probs: vec![1.0, 1.0] })
        );
    }

    #[test]
    fn test_parse_lineup_text_uses_last_block() {
        let text = "I'd put it in first normally.\n\n1st: Option C\n2nd: Option A\n3rd: Option B";
        assert_eq!(
            parse_lineup_text(text, 3),
            Some(LineupVerdict { ranking: vec![2, 0, 1], place_probs: vec![1.0, 1.0] })
        );
    }

    #[test]
    fn test_parse_lineup_text_no_marker_returns_none() {
        assert!(parse_lineup_text("no ranking here", 3).is_none());
    }

    #[test]
    fn test_logprob_parse_no_verdict_marker_discards() {
        // The judge is instructed to write "Verdict: Option <n>" — a bare
        // "Option 1" with no verdict marker anywhere must be discarded, not
        // scanned for.
        let logprobs = vec![
            plain(" Option"),
            LogprobContent { token: " 1".to_string(), top_logprobs: Some(vec![
                TopLogprob { token: "1".to_string(), logprob: -0.05 },
                TopLogprob { token: "2".to_string(), logprob: -3.5 },
            ]) },
            plain(" is"),
            plain(" better"),
        ];
        assert!(parse_response(&logprobs, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs.is_none());
    }

    /// Build a ranking letter token whose top_logprobs carry `probs` over the
    /// first `probs.len()` option letters. Generalizes [`letter_tok`] to any
    /// lineup size.
    fn letter_tok_n(tok: &str, probs: &[f64]) -> LogprobContent {
        let tlps = probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > 0.0)
            .map(|(i, &p)| TopLogprob {
                token: LINEUP_LETTERS[i].to_string(),
                logprob: p.ln(),
            })
            .collect();
        LogprobContent { token: tok.to_string(), top_logprobs: Some(tlps) }
    }

    /// A full ranking block for a lineup of `size`, ranked A, B, C, ... in
    /// order, where each slot puts half its conditional mass on the letter it
    /// emits and spreads the other half evenly over the still-unplaced ones.
    /// Every place's probability is then 0.5.
    fn halving_ranking_block(size: usize) -> Vec<LogprobContent> {
        const ORDINALS: [&str; MAX_LINEUP_SIZE] =
            ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th"];
        let mut lp = Vec::new();
        for rank in 0..size {
            if rank > 0 {
                lp.push(plain("\n"));
            }
            lp.push(plain(ORDINALS[rank]));
            lp.push(plain(":"));
            lp.push(plain(" Option"));
            let unplaced = size - rank;
            let mut dist = vec![0.0; size];
            dist[rank] = if unplaced == 1 { 1.0 } else { 0.5 };
            for slot in dist.iter_mut().take(size).skip(rank + 1) {
                *slot = 0.5 / (unplaced - 1) as f64;
            }
            lp.push(letter_tok_n(&format!(" {}", LINEUP_LETTERS[rank]), &dist));
        }
        lp
    }

    #[test]
    fn test_parse_lineup_place_probabilities_at_every_size() {
        // Each slot gives its pick half the still-unplaced mass, so every
        // place reads 0.5, and the unread last place adds no entry.
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let v = parse_lineup(&halving_ranking_block(size), size, 0.95)
                .unwrap_or_else(|| panic!("size {size}: should parse"));
            let ranking: Vec<usize> = (0..size).collect();
            assert_verdict(&v, &ranking, &vec![0.5; size - 1]);
        }
    }

    #[test]
    fn test_parse_lineup_ignores_the_last_slots_distribution() {
        // The final line must be present (a ranking is only valid if it places
        // every option) but carries no free information: only one option is
        // left for it, so scrambling its logprobs changes nothing.
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let block = halving_ranking_block(size);
            let expected = parse_lineup(&block, size, 0.95)
                .unwrap_or_else(|| panic!("size {size}: should parse"));

            let mut scrambled = halving_ranking_block(size);
            let last = scrambled.len() - 1;
            let mut junk = vec![0.5 / (size - 1) as f64; size];
            junk[size - 1] = 0.5;
            junk.reverse();
            scrambled[last] = letter_tok_n(&format!(" {}", LINEUP_LETTERS[size - 1]), &junk);

            let actual = parse_lineup(&scrambled, size, 0.95)
                .unwrap_or_else(|| panic!("size {size}: scrambled block should still parse"));
            assert_eq!(expected, actual, "size {size}: the last line's logprobs were read");
        }
    }

    #[test]
    fn test_lineup_letters_are_size_aware() {
        // A letter beyond the lineup is not an option letter — this is what
        // keeps the English words "A" and "I" from being read as options in
        // lineups too small to contain them.
        assert_eq!(lineup_letter_to_index('C', 3), Some(2));
        assert_eq!(lineup_letter_to_index('D', 3), None);
        assert_eq!(lineup_letter_to_index('D', 4), Some(3));
        assert_eq!(lineup_letter_to_index('I', 8), None);
        assert_eq!(lineup_letter_to_index('I', 9), Some(8));
        assert_eq!(lineup_letter_to_index('i', 9), None);
        assert_eq!(lineup_letter_to_index('a', 3), None);
    }

    #[test]
    fn test_parse_lineup_ignores_lowercase_letter_alternatives() {
        // At 1st place the top_logprobs put 0.02 on lowercase " a" (prose like
        // "Option a bit..."). It must not count as Option A: coverage is
        // 0.9 + 0.03 + 0.05 = 0.98 and A's share is 0.9 / 0.98, not 0.92 / 1.0.
        let mut lp = lineup_logprobs();
        lp[3] = LogprobContent {
            token: " A".to_string(),
            top_logprobs: Some(vec![
                TopLogprob { token: "A".to_string(), logprob: (0.9_f64).ln() },
                TopLogprob { token: "B".to_string(), logprob: (0.03_f64).ln() },
                TopLogprob { token: "C".to_string(), logprob: (0.05_f64).ln() },
                TopLogprob { token: " a".to_string(), logprob: (0.02_f64).ln() },
            ]),
        };
        let v = parse_lineup(&lp, 3, 0.95).expect("uppercase mass clears coverage");
        assert_verdict(&v, &[0, 1, 2], &[0.9 / 0.98, 0.8]);

        // With most of the mass on lowercase " a", uppercase coverage is only
        // 0.53 and the judgement must fail rather than read " a" as A.
        lp[3] = LogprobContent {
            token: " A".to_string(),
            top_logprobs: Some(vec![
                TopLogprob { token: "A".to_string(), logprob: (0.5_f64).ln() },
                TopLogprob { token: " a".to_string(), logprob: (0.47_f64).ln() },
                TopLogprob { token: "B".to_string(), logprob: (0.01_f64).ln() },
                TopLogprob { token: "C".to_string(), logprob: (0.02_f64).ln() },
            ]),
        };
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_logprob_parse_does_not_flip_on_second_option_after_verdict() {
        // "Verdict: Option 1, not Option 2": the second "Option 2" is inside
        // the old 10-token window and would overwrite Option 1 with Option 2
        // at ~99.9% confidence. Must read only Option 1.
        let lp = vec![
            plain("Verdict"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " 1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: (0.9_f64).ln() },
                    TopLogprob { token: "2".to_string(), logprob: (0.1_f64).ln() },
                ]),
            },
            plain(","), plain(" not"), plain(" Option"),
            LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "2".to_string(), logprob: (0.999_f64).ln() },
                    TopLogprob { token: "1".to_string(), logprob: (0.001_f64).ln() },
                ]),
            },
        ];
        let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE)
            .category_probs
            .expect("should parse Option 1");
        assert!((probs[0] - 0.9).abs() < 1e-6, "expected P(Option 1) ~0.9, got {:?}", probs);
    }

    #[test]
    fn test_logprob_parse_low_coverage_final_verdict_fails_without_fallback() {
        // An earlier valid Verdict: Option 1 followed by a final Verdict: Option 2
        // that fails coverage must fail the whole judgement rather than falling
        // back to the earlier verdict.
        let lp = vec![
            plain("Verdict"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " 1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: (0.98_f64).ln() },
                    TopLogprob { token: "2".to_string(), logprob: (0.02_f64).ln() },
                ]),
            },
            plain("\n"), plain("Verdict"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " 2".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "2".to_string(), logprob: (0.40_f64).ln() },
                    TopLogprob { token: " the".to_string(), logprob: (0.60_f64).ln() },
                ]),
            },
        ];
        assert!(parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs.is_none());
    }

    #[test]
    fn test_logprob_parse_rejects_prose_between_verdict_and_option() {
        let lp = vec![
            plain("My"), plain(" verdict"), plain(" is"), plain(" that"), plain(" Option"),
            LogprobContent {
                token: " 1".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "1".to_string(), logprob: (0.95_f64).ln() },
                    TopLogprob { token: "2".to_string(), logprob: (0.05_f64).ln() },
                ]),
            },
        ];
        assert!(parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs.is_none());
    }

    #[test]
    fn test_parse_lineup_ignores_trailing_prose_with_lowercase_a() {
        // Prose like "this option has a clear edge" or "this option a winner"
        // must not inject a bogus Option A slot after the real ranking block.
        let mut lp = lineup_logprobs();
        lp.extend([
            plain("\n"), plain("This"), plain(" option"), plain(" has"),
            letter_tok(" a", 0.99, 0.005, 0.005),
            plain(" clear"), plain(" edge"), plain("."),
            plain(" That"), plain(" option"),
            letter_tok(" a", 0.99, 0.005, 0.005),
            plain(" winner"), plain("."),
        ]);
        let v = parse_lineup(&lp, 3, 0.95).expect("trailing prose must not shift the ranking block");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }

    #[test]
    fn test_parse_lineup_coverage_checks_unplaced_mass_only() {
        // 1st place is Option A. At 2nd place, the model puts 0.90 mass on
        // repeating A and only 0.04 on B + 0.02 on C. Total option mass is
        // 0.96, but unplaced mass (B + C) is only 0.06 — must fail the 0.95
        // coverage gate.
        let lp = vec![
            plain("1st"), plain(":"), plain(" Option"),
            letter_tok(" A", 0.9, 0.03, 0.07),
            plain("\n"),
            plain("2nd"), plain(":"), plain(" Option"),
            LogprobContent {
                token: " B".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: "A".to_string(), logprob: (0.90_f64).ln() },
                    TopLogprob { token: "B".to_string(), logprob: (0.04_f64).ln() },
                    TopLogprob { token: "C".to_string(), logprob: (0.02_f64).ln() },
                    TopLogprob { token: " the".to_string(), logprob: (0.04_f64).ln() },
                ]),
            },
            plain("\n"),
            plain("3rd"), plain(":"), plain(" Option"),
            letter_tok(" C", 0.0, 0.0, 1.0),
        ];
        assert!(parse_lineup(&lp, 3, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_word_boundaries_bold_and_newlines() {
        // Leading word-boundary rejects "description A" / "adoption B"; newline
        // between "option" and "A" is rejected; bold "**Option** **A**" works.
        let text = "1st: **Option** **B**\n2nd: Option **A**\n3rd: Option C\n\
                    Matches description A and adoption B. That is the best option\nA closing note.";
        assert_eq!(
            parse_lineup_text(text, 3),
            Some(LineupVerdict { ranking: vec![1, 0, 2], place_probs: vec![1.0, 1.0] })
        );

        let lp = vec![
            plain("1st"), plain(":"), plain(" **"), plain("Option"), plain("**"), plain(" **"),
            letter_tok("A", 0.9, 0.03, 0.07), plain("**"), plain("\n"),
            plain("2nd"), plain(":"), plain(" Option"), plain(" **"),
            letter_tok("B", 0.0, 0.8, 0.2), plain("**"), plain("\n"),
            plain("3rd"), plain(":"), plain(" Option"),
            letter_tok(" C", 0.0, 0.0, 1.0), plain("\n"),
            plain("description"), letter_tok(" A", 0.99, 0.005, 0.005), plain("\n"),
            plain("best"), plain(" option"), plain("\n"),
            letter_tok("A", 0.99, 0.005, 0.005), plain(" note"),
        ];
        let v = parse_lineup(&lp, 3, 0.95).expect("bold block should survive trailing prose");
        assert_verdict(&v, &[0, 1, 2], &[0.9, 0.8]);
    }
}
