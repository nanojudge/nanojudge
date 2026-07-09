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
    // No word-boundary requirement: a spurious match inside a longer word is
    // harmless because a marker only counts if a clean, high-coverage letter
    // token follows it.
    let marker_starts = marker_scan_starts(&tokens, "verdict", false);

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
        if !is_upper && !has_colon && first_char.eq_ignore_ascii_case(&'a') {
            continue;
        }

        let top_logprobs = match &logprobs[i].top_logprobs {
            Some(tlps) if !tlps.is_empty() => tlps,
            _ => return None,
        };

        let mut choice_probs = [0.0_f64; 4];

        for tlp in top_logprobs {
            let clean = tlp.token.trim().trim_end_matches(':');
            if clean.len() == 1
                && let Some(tidx) = letter_to_index(clean.chars().next().unwrap())
            {
                choice_probs[tidx] += tlp.logprob.exp();
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

// ---------------------------------------------------------------------------
// Three-way (3-item) parsing
// ---------------------------------------------------------------------------

/// The three option letters of a 3-way comparison, in order (A, B, C).
const THREE_WAY_LETTERS: [char; 3] = ['A', 'B', 'C'];

fn three_way_letter_to_index(c: char) -> Option<usize> {
    THREE_WAY_LETTERS.iter().position(|&l| l == c.to_ascii_uppercase())
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
        let idx = match three_way_letter_to_index(first_char) {
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
                && let Some(tidx) = three_way_letter_to_index(clean.chars().next().unwrap())
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

/// Extract the 1st- and 2nd-rank slots from a three-way response's logprobs.
///
/// The ranking block is written as three "Option <letter>" lines at the end
/// (with or without "Nth:" prefixes). We collect every "Option <letter>" whose
/// letter reads cleanly and take the LAST three — the trailing block — so earlier
/// "Option X" mentions in the analysis prose are ignored. Those three emitted
/// letters must be a full 1-2-3 ranking (a permutation of A/B/C). If we can't get
/// three clean results, or they aren't all distinct (e.g. the model repeated an
/// option), the whole comparison is thrown out rather than filled in. Returns
/// `(first_slot, second_slot)`.
fn extract_three_way_slots(
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

/// Fold a three-way response's logprobs into a winner-distribution
/// `[q_A, q_B, q_C]` — the probability each option is the best of the three.
///
/// Keeps the 1st-place probability of the emitted winner, then splits the
/// residual between the other two by the *2nd-place* ratio (the focused
/// discrimination), discarding the noisy 1st-place tail between them. Returns
/// None if either rank slot fails to parse or clear the coverage threshold.
pub fn parse_three_way(logprobs: &[LogprobContent], min_logprob_coverage: f64) -> Option<[f64; 3]> {
    let (first, second) = extract_three_way_slots(logprobs, min_logprob_coverage)?;

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
        // the residual. Treat as an unparseable comparison rather than guessing.
        return None;
    }

    let mut q = [0.0_f64; 3];
    q[winner] = q_winner;
    q[x] = residual * (sx / denom);
    q[y] = residual * (sy / denom);
    Some(q)
}

/// Parse a three-way ranking from response text (--no-logprobs mode).
///
/// Same rule as the logprob path: take the LAST three "Option <letter>" mentions
/// (the trailing ranking block), require them to be a distinct 1-2-3 ranking, and
/// return a one-hot winner-distribution on the 1st. Without logprobs there is no
/// soft information, so the 2nd-vs-3rd ordering is dropped — only the winner is
/// kept. Returns None if three distinct results can't be read (throw it out).
pub fn parse_three_way_text(text: &str) -> Option<[f64; 3]> {
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
            && let Some(idx) = three_way_letter_to_index(c)
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
    fn test_logprob_parse_deepseek_three_token_marker_split() {
        // Captured from a real deepseek-chat response (2026-07-08): the marker
        // arrives split as "V"+"erd"+"ict" — three tokens — and the letter is a
        // standalone " D" token whose top_logprobs hold the A-D mass. The old
        // hard-coded two-token split matching could never find this marker.
        let lp = vec![
            plain(" a"), plain(" cat"), plain(".\n\n"),
            plain("V"), plain("erd"), plain("ict"),
            LogprobContent {
                token: " D".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: " D".to_string(), logprob: -0.002 },
                    TopLogprob { token: " C".to_string(), logprob: -6.488 },
                    TopLogprob { token: " B".to_string(), logprob: -9.691 },
                    TopLogprob { token: " A".to_string(), logprob: -13.276 },
                    TopLogprob { token: ":".to_string(), logprob: -17.279 },
                ]),
            },
            plain(":"), plain(" Option"), plain(" "), plain("2"), plain(","), plain(" clearly"),
        ];
        let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
            .expect("DeepSeek V+erd+ict marker split must parse");
        assert!(probs[3] > 0.99, "P(D) {} should dominate", probs[3]);
    }

    #[test]
    fn test_logprob_parse_two_token_marker_splits() {
        // Every two-token split of "Verdict" must be found by the offset-based
        // marker search (the old code special-cased only Ver+dict and Verd+ict).
        for (a, b) in [("V", "erdict"), ("Ve", "rdict"), ("Ver", "dict"), ("Verd", "ict"), ("Verdi", "ct"), ("Verdic", "t")] {
            let lp = vec![
                plain("Analysis"), plain(".\n\n"),
                plain(a), plain(b),
                LogprobContent {
                    token: " B".to_string(),
                    top_logprobs: Some(vec![
                        TopLogprob { token: "B".to_string(), logprob: -0.05 },
                        TopLogprob { token: "A".to_string(), logprob: -3.5 },
                        TopLogprob { token: "C".to_string(), logprob: -4.0 },
                        TopLogprob { token: "D".to_string(), logprob: -5.0 },
                    ]),
                },
            ];
            let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
                .unwrap_or_else(|| panic!("split {a}+{b} must parse"));
            assert!(probs[1] > 0.9, "split {a}+{b}: P(B) {} should dominate", probs[1]);
        }
    }

    #[test]
    fn test_logprob_parse_any_tokenization_of_marker() {
        // Simulate arbitrary tokenizers: chunk the response text into
        // pseudo-random 1-4 byte tokens and require the parse to succeed for
        // every chunking. The verdict letter itself stays a standalone token —
        // that is a hard requirement of logprobs mode (the letter token's
        // top_logprobs ARE the data), not something the parser can recover from.
        let prefix = "The analysis weighs both options and reaches a verdict on merit.\n\nVerdict";
        let suffix = ": Option 2, clearly";
        for seed in 0..200u64 {
            let mut lp = chunk_plain(prefix, seed);
            lp.push(LogprobContent {
                token: " D".to_string(),
                top_logprobs: Some(vec![
                    TopLogprob { token: " D".to_string(), logprob: -0.01 },
                    TopLogprob { token: " C".to_string(), logprob: -5.0 },
                    TopLogprob { token: " B".to_string(), logprob: -6.0 },
                    TopLogprob { token: " A".to_string(), logprob: -7.0 },
                ]),
            });
            lp.extend(chunk_plain(suffix, seed ^ 0x9e3779b9));
            let probs = parse_response(&lp, DEFAULT_MIN_LOGPROB_COVERAGE).category_probs
                .unwrap_or_else(|| panic!("seed {seed}: chunked tokenization must parse"));
            assert!(probs[3] > 0.9, "seed {seed}: P(D) {} should dominate", probs[3]);
        }
    }

    // --- Three-way parsing tests ---

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

    /// A full three-way ranking block: 1st=A (0.9/0.03/0.07), 2nd=B over the
    /// non-winners (B 0.8, C 0.2).
    fn three_way_logprobs() -> Vec<LogprobContent> {
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
    fn test_parse_three_way_folds_winner_distribution() {
        let q = parse_three_way(&three_way_logprobs(), 0.95).expect("should parse");
        // Winner A keeps its 1st-place prob; residual 0.10 split 0.8:0.2 → 0.08, 0.02.
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_three_way_split_ordinal_tokens() {
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
        let q = parse_three_way(&lp, 0.95).expect("split-ordinal tokens should parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
    }

    #[test]
    fn test_parse_three_way_split_option_marker() {
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
        let q = parse_three_way(&lp, 0.95).expect("split Option anchors should parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
    }

    #[test]
    fn test_parse_three_way_prose_options_word_is_not_an_anchor() {
        // Trailing prose containing "options" followed by a letter-like token
        // must not register as a fourth anchor and shift the ranking block.
        let mut lp = three_way_logprobs();
        lp.extend([plain("\n"), plain("Best"), plain(" of"), plain(" the"), plain(" options")]);
        lp.push(letter_tok(" A", 0.9, 0.05, 0.05));
        let q = parse_three_way(&lp, 0.95).expect("block before the prose must parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {} — prose 'options' shifted the block", q[0]);
    }

    #[test]
    fn test_parse_three_way_any_tokenization_of_markers() {
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
            let q = parse_three_way(&lp, 0.95)
                .unwrap_or_else(|| panic!("seed {seed}: chunked tokenization must parse"));
            assert!((q[0] - 0.9).abs() < 1e-9, "seed {seed}: q_A = {}", q[0]);
        }
    }

    #[test]
    fn test_parse_three_way_rejects_repeated_option() {
        // Model repeated the same option ("Option B" x3) instead of ranking — not
        // a distinct 1-2-3 ranking, so the whole comparison is thrown out.
        let lp = vec![
            plain(" Option"), letter_tok(" B", 0.02, 0.97, 0.01), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.02, 0.97, 0.01), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.02, 0.97, 0.01),
        ];
        assert!(parse_three_way(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_three_way_text_rejects_repeated_option() {
        assert!(parse_three_way_text("analysis\n\nOption B\nOption B\nOption B").is_none());
    }

    #[test]
    fn test_parse_three_way_rejects_fewer_than_three() {
        // Only two clean results → cannot form a full ranking → throw out.
        let lp = vec![
            plain(" Option"), letter_tok(" A", 0.9, 0.05, 0.05), plain("\n"),
            plain(" Option"), letter_tok(" B", 0.05, 0.9, 0.05),
        ];
        assert!(parse_three_way(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_three_way_bare_option_lines() {
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
        let q = parse_three_way(&lp, 0.95).expect("bare Option lines should parse");
        assert!((q[0] - 0.9).abs() < 1e-9, "q_A = {}", q[0]);
        assert!((q[1] - 0.08).abs() < 1e-9, "q_B = {}", q[1]);
        assert!((q[2] - 0.02).abs() < 1e-9, "q_C = {}", q[2]);
    }

    #[test]
    fn test_parse_three_way_ignores_prose_place_before_block() {
        // "in the first place" style prose (an earlier "first" ordinal) must not
        // derail the block — the LAST "1st"/"first" marker wins.
        let mut lp = vec![plain("in"), plain(" the"), plain(" first"), plain(" place"), plain(".")];
        lp.extend(three_way_logprobs());
        let q = parse_three_way(&lp, 0.95).expect("should still find the final block");
        assert!((q[0] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_parse_three_way_low_coverage_fails() {
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
        assert!(parse_three_way(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_three_way_missing_second_place_fails() {
        let lp = vec![
            plain("1st"), plain(":"), plain(" Option"),
            letter_tok(" A", 0.9, 0.05, 0.05),
        ];
        assert!(parse_three_way(&lp, 0.95).is_none());
    }

    #[test]
    fn test_parse_three_way_text_one_hot_on_winner() {
        let text = "Analysis here.\n\n1st: Option B\n2nd: Option A\n3rd: Option C";
        assert_eq!(parse_three_way_text(text), Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_parse_three_way_text_uses_last_block() {
        let text = "I'd put it in first normally.\n\n1st: Option C\n2nd: Option A\n3rd: Option B";
        assert_eq!(parse_three_way_text(text), Some([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_parse_three_way_text_no_marker_returns_none() {
        assert!(parse_three_way_text("no ranking here").is_none());
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
