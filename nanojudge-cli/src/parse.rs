/// Verdict extraction for LLM judgements.
///
/// The verdict line is "Verdict: Option 1" or "Verdict: Option 2". Two
/// separate parsing modes:
/// - Logprob mode: extracts continuous probabilities from the option-digit
///   token's logprobs.
/// - Text mode (--no-logprobs): extracts the discrete verdict from response text.
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

/// Find every occurrence of the marker word `needle` (lowercase ASCII) in the
/// token stream, regardless of how the tokenizer split it, and return the token
/// index just past the token holding the end of each occurrence — where the
/// scan for the following letter should start.
///
/// Works on the concatenated token text with a byte-offset table mapping match
/// positions back to token indices, so it is tokenization-agnostic: the marker
/// can arrive whole ("Verdict"), split in two ("Ver"+"dict"), or in three
/// ("V"+"erd"+"ict", as DeepSeek emits it). ASCII lowercasing keeps byte
/// offsets aligned with the original text.
///
/// `word_boundary` requires the match to not butt up against ASCII letters on
/// either side, so e.g. "options"/"optional" don't count as "option".
fn marker_scan_starts(tokens: &[&str], needle: &str, word_boundary: bool) -> Vec<usize> {
    // Concatenated token text plus each token's exclusive end offset into it.
    let mut text = String::new();
    let mut token_ends = Vec::with_capacity(tokens.len());
    for t in tokens {
        text.push_str(t);
        token_ends.push(text.len());
    }
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();

    let mut starts = Vec::new();
    let mut from = 0;
    while let Some(rel) = lower[from..].find(needle) {
        let begin = from + rel;
        let end = begin + needle.len();
        from = begin + 1;
        if word_boundary {
            let letter_before = begin > 0 && bytes[begin - 1].is_ascii_alphabetic();
            let letter_after = end < bytes.len() && bytes[end].is_ascii_alphabetic();
            if letter_before || letter_after {
                continue;
            }
        }
        // The token holding the match's last byte; the letter scan starts
        // after it.
        let tok_idx = token_ends.partition_point(|&e| e < end);
        starts.push(tok_idx + 1);
    }
    starts
}

/// Extract the judge's verdict distribution `[P(Option 1), P(Option 2)]` from
/// logprobs. Returns None if no "Verdict: Option <digit>" line is present,
/// extraction fails, or the digit mass is below the coverage threshold.
fn extract_pairwise_probabilities(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> Option<[f64; 2]> {
    if logprobs.is_empty() {
        return None;
    }

    let tokens: Vec<&str> = logprobs.iter().map(|lp| lp.token.as_str()).collect();

    // Collect every "Verdict" marker position. The analysis prose can contain
    // the word "verdict" (e.g. "reach a fair verdict") before or after the
    // real "Verdict: Option <digit>" line, so we attempt extraction after each
    // marker and keep the LAST successful one. Mirrors parse_response_text.
    // No marker at all means the judge ignored the required format — discard
    // rather than scan the response start for stray digits.
    // No word-boundary requirement on "verdict": a spurious match inside a
    // longer word is harmless because a marker only counts if an "Option"
    // anchor and a clean, high-coverage digit token follow it.
    let verdict_starts = marker_scan_starts(&tokens, "verdict", false);
    // The "Option" anchors, word-boundary-required so prose "options"/"optional"
    // don't count (same rule as the three-item lineup parser).
    let option_starts = marker_scan_starts(&tokens, "option", true);

    let mut result = None;
    for &vstart in &verdict_starts {
        for &ostart in &option_starts {
            if ostart < vstart {
                continue;
            }
            // The "Option" anchor must sit right after the verdict marker
            // (": Option" is at most ~8 single-byte tokens under any split).
            if ostart > vstart + 10 {
                break;
            }
            if let Some(probs) = extract_digit_after(logprobs, &tokens, ostart, min_logprob_coverage) {
                result = Some(probs);
            }
        }
    }
    result
}

/// Scan the few tokens following one "Option" anchor for the option digit and
/// build the `[P(1), P(2)]` distribution from its top_logprobs. Returns None
/// if no digit token is found or the digit mass is below the coverage threshold.
fn extract_digit_after(
    logprobs: &[LogprobContent],
    tokens: &[&str],
    search_start: usize,
    min_logprob_coverage: f64,
) -> Option<[f64; 2]> {
    let search_end = (search_start + 6).min(tokens.len());

    for i in search_start..search_end {
        let tok = tokens[i].trim();
        if tok.is_empty() {
            continue;
        }

        let first_char = tok.chars().next().unwrap();
        if digit_to_index(first_char).is_none() {
            continue;
        }

        // The digit must stand alone (allow a trailing colon/period), so a
        // multi-digit number in prose is never read as a verdict.
        let rest = &tok[first_char.len_utf8()..];
        if !rest.is_empty() && rest != ":" && rest != "." {
            continue;
        }

        let top_logprobs = match &logprobs[i].top_logprobs {
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
        if prob_sum >= min_logprob_coverage {
            // Normalize into a proper distribution over the two options.
            for p in &mut choice_probs {
                *p /= prob_sum;
            }
            return Some(choice_probs);
        } else {
            // Logprobs don't cover enough of the digit space — parse fails.
            return None;
        }
    }

    None
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
// Three-item lineup (3-item) parsing
// ---------------------------------------------------------------------------

/// The three option letters of a three-item lineup judgement, in order (A, B, C).
const LINEUP_LETTERS: [char; 3] = ['A', 'B', 'C'];

fn lineup_letter_to_index(c: char) -> Option<usize> {
    LINEUP_LETTERS.iter().position(|&l| l == c.to_ascii_uppercase())
}

/// A single rank slot's extracted distribution over the three option letters
/// `[P(A), P(B), P(C)]`, plus which letter the judge actually emitted there.
struct RankSlot {
    dist: [f64; 3],
    letter: usize,
}

/// Read the distribution over {A, B, C} from a ranking-line letter token's
/// top_logprobs. `search_start` is the token index just past an "Option" marker;
/// scans forward a few tokens for the single option letter and reads its
/// top_logprobs. Returns None if no letter token with usable top_logprobs is
/// found, or the A-C mass is below `min_logprob_coverage`.
fn extract_rank_slot(
    logprobs: &[LogprobContent],
    tokens: &[&str],
    search_start: usize,
    min_logprob_coverage: f64,
) -> Option<RankSlot> {
    let search_end = (search_start + 6).min(tokens.len());
    for i in search_start..search_end {
        let tok = tokens[i].trim();
        if tok.is_empty() {
            continue;
        }
        let first_char = tok.chars().next().unwrap();
        let idx = match lineup_letter_to_index(first_char) {
            Some(idx) => idx,
            None => continue,
        };
        // The letter must stand alone (allow a trailing colon/period), so "Aardvark"
        // or the word "Also" don't get read as option A.
        let rest = &tok[first_char.len_utf8()..];
        if !rest.is_empty() && rest != ":" && rest != "." {
            continue;
        }

        let top_logprobs = match &logprobs[i].top_logprobs {
            Some(tlps) if !tlps.is_empty() => tlps,
            _ => return None,
        };

        let mut dist = [0.0_f64; 3];
        for tlp in top_logprobs {
            let clean = tlp.token.trim().trim_end_matches([':', '.']);
            if clean.len() == 1
                && let Some(tidx) = lineup_letter_to_index(clean.chars().next().unwrap())
            {
                dist[tidx] += tlp.logprob.exp();
            }
        }
        let mass: f64 = dist.iter().sum();
        if mass < min_logprob_coverage {
            return None;
        }
        for p in &mut dist {
            *p /= mass;
        }
        return Some(RankSlot { dist, letter: idx });
    }
    None
}

/// Extract the 1st- and 2nd-rank slots from a three-item lineup response's logprobs.
///
/// The ranking block is written as three "Option <letter>" lines at the end
/// (with or without "Nth:" prefixes). We collect every "Option <letter>" whose
/// letter reads cleanly and take the LAST three — the trailing block — so earlier
/// "Option X" mentions in the analysis prose are ignored. Those three emitted
/// letters must be a full 1-2-3 ranking (a permutation of A/B/C). If we can't get
/// three clean results, or they aren't all distinct (e.g. the model repeated an
/// option), the whole judgement is thrown out rather than filled in. Returns
/// `(first_slot, second_slot)`.
fn extract_lineup_slots(
    logprobs: &[LogprobContent],
    min_logprob_coverage: f64,
) -> Option<(RankSlot, RankSlot)> {
    if logprobs.is_empty() {
        return None;
    }
    let tokens: Vec<&str> = logprobs.iter().map(|lp| lp.token.as_str()).collect();

    // Anchor on "Option" regardless of tokenization. Word boundaries required
    // so prose "options"/"optional" don't become anchors that inject stray
    // letter reads into the block.
    let mut slots: Vec<RankSlot> = Vec::new();
    for start in marker_scan_starts(&tokens, "option", true) {
        if let Some(slot) = extract_rank_slot(logprobs, &tokens, start, min_logprob_coverage) {
            slots.push(slot);
        }
    }

    // The ranking block is the last three results. Require all three, and require
    // them to be distinct A/B/C (a real 1-2-3 ranking) — otherwise throw it out.
    if slots.len() < 3 {
        return None;
    }
    let block = &slots[slots.len() - 3..];
    let mut seen = [false; 3];
    for s in block {
        seen[s.letter] = true;
    }
    if !seen.iter().all(|&b| b) {
        return None;
    }

    let first = RankSlot { dist: block[0].dist, letter: block[0].letter };
    let second = RankSlot { dist: block[1].dist, letter: block[1].letter };
    Some((first, second))
}

/// Fold a three-item lineup response's logprobs into a winner-distribution
/// `[q_A, q_B, q_C]` — the probability each option is the best of the three.
///
/// Keeps the 1st-place probability of the emitted winner, then splits the
/// residual between the other two by the *2nd-place* ratio (the focused
/// discrimination), discarding the noisy 1st-place tail between them. Returns
/// None if either rank slot fails to parse or clear the coverage threshold.
pub fn parse_lineup(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> Option<[f64; 3]> {
    let (first, second) = extract_lineup_slots(logprobs, min_logprob_coverage)?;

    let winner = first.letter;
    let q_winner = first.dist[winner];
    let residual = 1.0 - q_winner;

    // The two non-winners, split by the 2nd-place distribution restricted to them.
    let others: Vec<usize> = (0..3).filter(|&i| i != winner).collect();
    let (x, y) = (others[0], others[1]);
    let sx = second.dist[x];
    let sy = second.dist[y];
    let denom = sx + sy;
    if denom <= 0.0 {
        // The 2nd-place slot put no mass on either non-winner — no way to split
        // the residual. Treat as an unparseable judgement rather than guessing.
        return None;
    }

    let mut q = [0.0_f64; 3];
    q[winner] = q_winner;
    q[x] = residual * (sx / denom);
    q[y] = residual * (sy / denom);
    Some(q)
}

/// Parse a three-item lineup ranking from response text (--no-logprobs mode).
///
/// Same rule as the logprob path: take the LAST three "Option <letter>" mentions
/// (the trailing ranking block), require them to be a distinct 1-2-3 ranking, and
/// return a one-hot winner-distribution on the 1st. Without logprobs there is no
/// soft information, so the 2nd-vs-3rd ordering is dropped — only the winner is
/// kept. Returns None if three distinct results can't be read (throw it out).
pub fn parse_lineup_text(text: &str) -> Option<[f64; 3]> {
    let lower = text.to_ascii_lowercase();
    let bytes = text.as_bytes();

    // Every uppercase, standalone "Option <letter>" mention, in order.
    let mut letters: Vec<usize> = Vec::new();
    let mut search = 0;
    while let Some(rel) = lower[search..].find("option") {
        let mut pos = search + rel + "option".len();
        while pos < text.len() && matches!(bytes[pos], b' ' | b'\t' | b':') {
            pos += 1;
        }
        if let Some(c) = text[pos..].chars().next()
            && let Some(idx) = lineup_letter_to_index(c)
            && c.is_ascii_uppercase()
        {
            // Must stand alone: next char is not alphanumeric (so "Options" is not O/A).
            let next = text[pos + c.len_utf8()..].chars().next();
            if next.is_none_or(|nc| !nc.is_alphanumeric()) {
                letters.push(idx);
            }
        }
        search += rel + "option".len();
    }

    if letters.len() < 3 {
        return None;
    }
    let block = &letters[letters.len() - 3..];
    let mut seen = [false; 3];
    for &l in block {
        seen[l] = true;
    }
    if !seen.iter().all(|&b| b) {
        return None;
    }

    let mut q = [0.0_f64; 3];
    q[block[0]] = 1.0; // 1st place
    Some(q)
}

/// Parse a verdict from response text (for --no-logprobs mode).
///
/// Finds the last "Verdict: Option <digit>" in the text and returns a one-hot
/// distribution over the two options. Uses the last occurrence to handle cases
/// where "verdict" appears in the analysis before the final verdict.
pub fn parse_response_text(text: &str) -> ParseResult {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut result = None;
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
                    result = Some(idx);
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

    #[test]
    fn test_parse_lineup_folds_winner_distribution() {
        let q = parse_lineup(&lineup_logprobs(), 0.95).expect("should parse");
        // Winner A keeps its 1st-place prob; residual 0.10 split 0.8:0.2 → 0.08, 0.02.
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-9);
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
        let q = parse_lineup(&lp, 0.95).expect("split-ordinal tokens should parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
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
        let q = parse_lineup(&lp, 0.95).expect("split Option anchors should parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
    }

    #[test]
    fn test_parse_lineup_prose_options_word_is_not_an_anchor() {
        // Trailing prose containing "options" followed by a letter-like token
        // must not register as a fourth anchor and shift the ranking block.
        let mut lp = lineup_logprobs();
        lp.extend([plain("\n"), plain("Best"), plain(" of"), plain(" the"), plain(" options")]);
        lp.push(letter_tok(" A", 0.9, 0.05, 0.05));
        let q = parse_lineup(&lp, 0.95).expect("block before the prose must parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {} — prose 'options' shifted the block", q[0]);
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
            let q = parse_lineup(&lp, 0.95)
                .unwrap_or_else(|| panic!("seed {seed}: chunked tokenization must parse"));
            assert!((q[0] - 0.9).abs() < 1e-9, "seed {seed}: q_A = {}", q[0]);
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
        assert!(parse_lineup(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_text_rejects_repeated_option() {
        assert!(parse_lineup_text("analysis\n\nOption B\nOption B\nOption B").is_none());
    }

    #[test]
    fn test_parse_lineup_rejects_fewer_than_three() {
        // Only two clean results → cannot form a full ranking → throw out.
        let lp = vec![
            plain(" Option"), letter_tok(" A", 0.9, 0.05, 0.05), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.05, 0.9, 0.05),
        ];
        assert!(parse_lineup(&lp, 0.95).is_none());
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
        let q = parse_lineup(&lp, 0.95).expect("bare Option lines should parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
    }

    #[test]
    fn test_parse_lineup_ignores_prose_place_before_block() {
        // "in the first place" style prose (an earlier "first" ordinal) must not
        // derail the block — the LAST "1st"/"first" marker wins.
        let mut lp = vec![plain("in"), plain(" the"), plain(" first"), plain(" place"), plain(".")];
        lp.extend(lineup_logprobs());
        let q = parse_lineup(&lp, 0.95).expect("should still find the final block");
        assert!((q[0] - 0.9).abs() < 1e-9);
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
        assert!(parse_lineup(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_missing_second_place_fails() {
        let lp = vec![
            plain("1st"), plain(":"), plain(" Option"),
            letter_tok(" A", 0.9, 0.05, 0.05),
        ];
        assert!(parse_lineup(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_lineup_text_one_hot_on_winner() {
        let text = "Analysis here.\n\n1st: Option B\n2nd: Option A\n3rd: Option C";
        assert_eq!(parse_lineup_text(text), Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_parse_lineup_text_uses_last_block() {
        let text = "I'd put it in first normally.\n\n1st: Option C\n2nd: Option A\n3rd: Option B";
        assert_eq!(parse_lineup_text(text), Some([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_parse_lineup_text_no_marker_returns_none() {
        assert!(parse_lineup_text("no ranking here").is_none());
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
}
