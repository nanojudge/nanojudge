//! Prompt building for LLM judgements.
//!
//! Supports custom prompt templates with variable substitution.
//! Pairwise templates: $criterion, $option1 and $option2 are required;
//! $name1, $name2 (item titles) and $length (the analysis-length setting)
//! are optional. Lineup templates: $criterion and one $option<letter> per
//! item ($optionA through $optionI) are required; $length is optional.
//!
//! If no template is provided, a sensible default is used that produces
//! a "Verdict: Option <n>" line (pairwise) or one "<Nth> place is Option
//! <letter>" line per item (lineup).

use crate::bail;
use nanojudge_core::constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE};

pub const DEFAULT_TEMPLATE: &str = "\
$criterion

Option 1:
$option1

Option 2:
$option2

Instructions:
Write an analysis ($length). Analyse both options before forming a preference. You MUST end your response with one of these lines verbatim:

Verdict: Option 1
Verdict: Option 2
";

pub const DEFAULT_TEMPLATE_NO_REASONING: &str = "\
$criterion

Option 1:
$option1

Option 2:
$option2

Instructions:
Respond only with one of these lines verbatim:

Verdict: Option 1
Verdict: Option 2
";

const REQUIRED_VARIABLES: &[&str] = &["$criterion", "$option1", "$option2"];
const REQUIRED_VARIABLES_NO_REASONING: &[&str] = &["$criterion", "$option1", "$option2"];

// --- Lineup judgement templates ---
//
// A lineup template names its options literally: `$optionA`, `$optionB`, ...,
// one per item, up to `$optionI` for a nine-item lineup. A template is therefore
// written for one specific lineup size; using it at another size is an error
// rather than something the tool papers over.
//
// The judge ranks every option (first place through last); the parser reads the
// trailing "Option <letter>" lines and folds them into a winner-distribution
// that feeds `winner_dist_to_edges`.

/// Option letters, in presentation order. Index 0 is slot A.
pub const OPTION_LETTERS: [char; MAX_LINEUP_SIZE] =
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];

/// Ordinal words for the ranking lines, one per rank position.
const ORDINALS: [&str; MAX_LINEUP_SIZE] = [
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
];

/// The `$option` variable name for slot `index` (`$optionA`, `$optionB`, ...).
fn option_variable(index: usize) -> String {
    format!("$option{}", OPTION_LETTERS[index])
}

/// Every variable a `lineup_size` lineup template must contain: `$criterion`
/// plus one `$option<letter>` per item.
fn required_lineup_variables(lineup_size: usize) -> Vec<String> {
    let mut vars = vec!["$criterion".to_string()];
    vars.extend((0..lineup_size).map(option_variable));
    vars
}

/// The ranking-instruction block: one "Nth place is Option X" line per rank,
/// with the placeholder letters the judge is told to replace.
fn ranking_lines(lineup_size: usize) -> String {
    // Placeholders start well past the option letters so they can't be mistaken
    // for a real option: X, Y, Z, then W, V, ... backwards.
    const PLACEHOLDERS: [char; MAX_LINEUP_SIZE] =
        ['X', 'Y', 'Z', 'W', 'V', 'U', 'T', 'S', 'R'];
    (0..lineup_size)
        .map(|i| format!("{} place is Option {}", ORDINALS[i], PLACEHOLDERS[i]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The option blocks: "Option A:\n$optionA\n\nOption B:\n$optionB\n\n...".
fn option_blocks(lineup_size: usize) -> String {
    (0..lineup_size)
        .map(|i| format!("Option {}:\n{}", OPTION_LETTERS[i], option_variable(i)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// A comma-separated list of the option letters, for prose ("A, B, or C").
fn option_letter_list(lineup_size: usize) -> String {
    let letters: Vec<String> = (0..lineup_size).map(|i| OPTION_LETTERS[i].to_string()).collect();
    match letters.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{}, or {}", rest.join(", "), last),
        _ => letters.join(", "),
    }
}

/// The lineup size written as an English word, as the instruction prose reads
/// better spelled out ("Analyse all three options") than in digits.
fn size_word(lineup_size: usize) -> &'static str {
    const WORDS: [&str; MAX_LINEUP_SIZE + 1] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    WORDS[lineup_size]
}

/// The built-in lineup template for `lineup_size` items, with an analysis step.
///
/// The 3-item form is the shape all sizes follow:
///
/// ```text
/// $criterion
///
/// Option A:
/// $optionA
/// ...
/// Instructions:
/// Write an analysis ($length). Analyse all three options before forming a
/// preference. You MUST end your response with these three lines, ...
/// ```
pub fn default_lineup_template(lineup_size: usize) -> String {
    format!(
        "{criterion}\n\n{blocks}\n\nInstructions:\nWrite an analysis ($length). \
Analyse all {count} options before forming a preference. You MUST end your response \
with these {count} lines, replacing the placeholder letters with the letter of the \
option ({letters}):\n\n{lines}\n",
        criterion = "$criterion",
        blocks = option_blocks(lineup_size),
        count = size_word(lineup_size),
        letters = option_letter_list(lineup_size),
        lines = ranking_lines(lineup_size),
    )
}

/// The built-in lineup template for `lineup_size` items, verdict only (no
/// reasoning step).
pub fn default_lineup_template_no_reasoning(lineup_size: usize) -> String {
    format!(
        "{criterion}\n\n{blocks}\n\nInstructions:\nRespond only with these {count} lines, \
replacing the placeholder letters with the letter of the option ({letters}):\n\n{lines}\n",
        criterion = "$criterion",
        blocks = option_blocks(lineup_size),
        count = size_word(lineup_size),
        letters = option_letter_list(lineup_size),
        lines = ranking_lines(lineup_size),
    )
}


/// Validate that a template contains all required variables.
/// Returns an error message listing any missing variables.
pub fn validate_template(template: &str, reasoning_enabled: bool) -> Result<(), String> {
    let required = if reasoning_enabled { REQUIRED_VARIABLES } else { REQUIRED_VARIABLES_NO_REASONING };
    let missing: Vec<&&str> = required
        .iter()
        .filter(|var| !template.contains(**var))
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Prompt template is missing required variable(s): {}",
            missing.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        ))
    }
}

/// Validate that a lineup template matches `lineup_size`: it must contain
/// `$criterion` and exactly one `$option<letter>` per item — `$optionA` through
/// `$optionC` for a 3-item lineup, through `$optionI` for a 9-item one.
///
/// A template carrying option variables beyond the lineup size was written for a
/// different size and is rejected, rather than silently leaving `$optionD` in the
/// prompt text sent to the judge.
pub fn validate_lineup_template(template: &str, lineup_size: usize) -> Result<(), String> {
    let missing: Vec<String> = required_lineup_variables(lineup_size)
        .into_iter()
        .filter(|var| !template.contains(var.as_str()))
        .collect();

    if !missing.is_empty() {
        return Err(format!(
            "Lineup prompt template for {lineup_size} items is missing required variable(s): {}",
            missing.join(", ")
        ));
    }

    let extra: Vec<String> = (lineup_size..MAX_LINEUP_SIZE)
        .map(option_variable)
        .filter(|var| template.contains(var.as_str()))
        .collect();

    if !extra.is_empty() {
        return Err(format!(
            "Lineup prompt template is for a larger lineup than lineup-size={lineup_size}: \
it uses {}. Use a template with exactly {} option variable(s), or change lineup-size.",
            extra.join(", "),
            lineup_size
        ));
    }

    Ok(())
}

/// Load a prompt template from a file path, validate it, and return the contents.
pub fn load_template(path: &std::path::Path, reasoning_enabled: bool) -> String {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| bail(format!("Failed to read prompt template {}: {e}", path.display())));

    if let Err(msg) = validate_template(&content, reasoning_enabled) {
        bail(format!("{} (in {})", msg, path.display()));
    }

    content
}

/// Load a lineup prompt template from a file path, validate it against the
/// lineup size, and return it.
pub fn load_lineup_template(path: &std::path::Path, lineup_size: usize) -> String {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| bail(format!("Failed to read prompt template {}: {e}", path.display())));

    if let Err(msg) = validate_lineup_template(&content, lineup_size) {
        bail(format!("{} (in {})", msg, path.display()));
    }

    content
}

/// Build a judgement prompt by substituting variables into a template.
///
/// Single-pass substitution: only variables in the TEMPLATE are replaced.
/// Substituted values are never rescanned, so item text containing literal
/// `$option2`, `$length`, etc. (common when ranking code) passes through
/// untouched instead of being recursively substituted.
pub fn build_prompt(template: &str, criterion: &str, option1: &str, option2: &str, name1: &str, name2: &str, analysis_length: &str) -> String {
    let vars: [(&str, &str); 6] = [
        ("$criterion", criterion),
        ("$option1", option1),
        ("$option2", option2),
        ("$name1", name1),
        ("$name2", name2),
        ("$length", analysis_length),
    ];

    let mut out = String::with_capacity(template.len() + option1.len() + option2.len());
    let mut rest = template;
    while let Some(pos) = rest.find('$') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if let Some((var, value)) = vars.iter().find(|(v, _)| tail.starts_with(v)) {
            out.push_str(value);
            rest = &tail[var.len()..];
        } else {
            out.push('$');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Build a lineup judgement prompt by substituting the options into a template.
/// Single-pass, like `build_prompt`: item text containing literal template
/// tokens passes through untouched.
///
/// `option_texts` is in presentation order — index 0 fills `$optionA`.
///
/// # Panics
///
/// Panics if `option_texts` is not a valid lineup size (2..=9).
pub fn build_lineup_prompt(
    template: &str,
    criterion: &str,
    option_texts: &[&str],
    analysis_length: &str,
) -> String {
    let lineup_size = option_texts.len();
    assert!(
        (MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE).contains(&lineup_size),
        "lineup size must be between {MIN_LINEUP_SIZE} and {MAX_LINEUP_SIZE}, got {lineup_size}"
    );

    // No variable name is a prefix of another, so substitution order does not
    // affect the result; options are listed first only to read in lineup order.
    let mut vars: Vec<(String, &str)> = (0..lineup_size)
        .map(|i| (option_variable(i), option_texts[i]))
        .collect();
    vars.push(("$criterion".to_string(), criterion));
    vars.push(("$length".to_string(), analysis_length));

    let options_len: usize = option_texts.iter().map(|o| o.len()).sum();
    let mut out = String::with_capacity(template.len() + options_len);
    let mut rest = template;
    while let Some(pos) = rest.find('$') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if let Some((var, value)) = vars.iter().find(|(v, _)| tail.starts_with(v.as_str())) {
            out.push_str(value);
            rest = &tail[var.len()..];
        } else {
            out.push('$');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lineup_template_spells_the_size_as_a_word() {
        // Instruction prose reads as English at every size, and never as a
        // digit that could be mistaken for part of the ranking format.
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            for template in
                [default_lineup_template(size), default_lineup_template_no_reasoning(size)]
            {
                let word = size_word(size);
                assert!(
                    template.contains(&format!("these {word} lines")),
                    "size {size}: line count not spelled out:\n{template}"
                );
                assert!(
                    !template.contains(&format!("these {size} lines")),
                    "size {size}: line count left as a digit:\n{template}"
                );
            }
            assert!(
                default_lineup_template(size).contains(&format!("all {} options", size_word(size))),
                "size {size}: option count not spelled out"
            );
        }
    }

    #[test]
    fn test_lineup_template_size_three_matches_the_historical_wording() {
        // The three-item template predates variable sizes. Its option blocks,
        // ranking lines and letter list are reproduced exactly; only the
        // sentence naming the placeholders is deliberately reworded, since
        // listing them individually does not scale to nine.
        let expected = "\
$criterion

Option A:
$optionA

Option B:
$optionB

Option C:
$optionC

Instructions:
Write an analysis ($length). Analyse all three options before forming a preference. \
You MUST end your response with these three lines, replacing the placeholder letters \
with the letter of the option (A, B, or C):

First place is Option X
Second place is Option Y
Third place is Option Z
";
        assert_eq!(default_lineup_template(3), expected);
    }

    use super::*;

    #[test]
    fn test_default_template_is_valid() {
        validate_template(DEFAULT_TEMPLATE, true).unwrap();
    }

    #[test]
    fn test_build_prompt_with_default_template() {
        let prompt = build_prompt(DEFAULT_TEMPLATE, "Which is tastier?", "Pizza", "Sushi", "pizza", "sushi", "2 paragraphs");
        assert!(prompt.starts_with("Which is tastier?"));
        assert!(prompt.contains("Option 1:\nPizza"));
        assert!(prompt.contains("Option 2:\nSushi"));
        assert!(prompt.contains("Write an analysis (2 paragraphs)."));
        assert!(prompt.contains("Verdict: Option 1"));
        assert!(prompt.contains("Verdict: Option 2"));
    }

    #[test]
    fn test_custom_template() {
        let template = "Compare $option1 vs $option2 for $criterion. Be $length.";
        let prompt = build_prompt(template, "taste", "Pizza", "Sushi", "pizza", "sushi", "brief");
        assert_eq!(prompt, "Compare Pizza vs Sushi for taste. Be brief.");
    }

    #[test]
    fn test_build_prompt_item_text_with_variable_tokens_passes_through() {
        // Item text containing template-variable tokens (common when ranking
        // code) must stay literal — not get recursively substituted.
        let template = "$criterion\n1: $option1\n2: $option2\nBe $length.";
        let prompt = build_prompt(
            template,
            "Which script is cleaner?",
            "echo $option2 and $length",
            "print('hi')",
            "", "",
            "brief",
        );
        assert_eq!(
            prompt,
            "Which script is cleaner?\n1: echo $option2 and $length\n2: print('hi')\nBe brief."
        );
    }

    #[test]
    fn test_build_prompt_unknown_dollar_token_kept() {
        let template = "$criterion costs $5 or $unknown\n$option1 vs $option2 ($length)";
        let prompt = build_prompt(template, "Cheaper?", "A", "B", "", "", "brief");
        assert_eq!(prompt, "Cheaper? costs $5 or $unknown\nA vs B (brief)");
    }

    #[test]
    fn test_validate_missing_variables() {
        let result = validate_template("Just $option1 and $option2", true);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("$criterion"));
    }

    #[test]
    fn test_validate_complete_template() {
        let template = "$criterion\n$option1\n$option2\n$length";
        validate_template(template, true).unwrap();
    }

    #[test]
    fn test_no_reasoning_template_has_no_length() {
        assert!(!DEFAULT_TEMPLATE_NO_REASONING.contains("$length"));
    }

    #[test]
    fn test_no_reasoning_template_has_verdicts() {
        assert!(DEFAULT_TEMPLATE_NO_REASONING.contains("Verdict: Option 1"));
        assert!(DEFAULT_TEMPLATE_NO_REASONING.contains("Verdict: Option 2"));
    }

    // --- Lineup template tests ---

    #[test]
    fn test_default_lineup_templates_valid_at_every_size() {
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            validate_lineup_template(&default_lineup_template(size), size)
                .unwrap_or_else(|e| panic!("size {size}: {e}"));
            validate_lineup_template(&default_lineup_template_no_reasoning(size), size)
                .unwrap_or_else(|e| panic!("size {size} (no reasoning): {e}"));
        }
    }

    #[test]
    fn test_build_lineup_prompt() {
        let prompt = build_lineup_prompt(
            &default_lineup_template(3),
            "Which is tastier?",
            &["Pizza", "Sushi", "Tacos"],
            "2 paragraphs",
        );
        assert!(prompt.starts_with("Which is tastier?"));
        assert!(prompt.contains("Option A:\nPizza"));
        assert!(prompt.contains("Option B:\nSushi"));
        assert!(prompt.contains("Option C:\nTacos"));
        assert!(prompt.contains("First place is Option X"));
        assert!(prompt.contains("Third place is Option Z"));
        assert!(prompt.contains("Write an analysis (2 paragraphs)."));
    }

    /// Every option's text lands in its own slot, and no `$option` variable is
    /// left unsubstituted, at any lineup size.
    #[test]
    fn test_build_lineup_prompt_every_size() {
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let texts: Vec<String> = (0..size).map(|i| format!("item-text-{i}")).collect();
            let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
            let prompt = build_lineup_prompt(
                &default_lineup_template(size),
                "Which is best?",
                &refs,
                "2 paragraphs",
            );
            for (i, text) in texts.iter().enumerate() {
                assert!(
                    prompt.contains(&format!("Option {}:\n{}", OPTION_LETTERS[i], text)),
                    "size {size}: option {i} missing from prompt"
                );
            }
            assert!(!prompt.contains("$option"), "size {size}: unsubstituted variable");
            assert!(!prompt.contains("$criterion"), "size {size}: unsubstituted criterion");
            assert!(!prompt.contains("$length"), "size {size}: unsubstituted length");
            // One ranking line per rank.
            assert!(prompt.contains("First place is Option X"), "size {size}");
            assert!(
                prompt.contains(&format!("{} place is Option", ORDINALS[size - 1])),
                "size {size}: last ranking line missing"
            );
        }
    }

    /// The ranking-line placeholders must never collide with a real option
    /// letter, or the parser would read them as part of the block.
    #[test]
    fn test_ranking_placeholders_are_not_option_letters() {
        for size in MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE {
            let lines = ranking_lines(size);
            for letter in &OPTION_LETTERS[..size] {
                assert!(
                    !lines.contains(&format!("Option {letter}")),
                    "size {size}: placeholder collides with option {letter}"
                );
            }
        }
    }

    #[test]
    fn test_validate_lineup_missing_option() {
        let result = validate_lineup_template("$criterion $optionA $optionB only", 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("$optionC"));
    }

    /// A template written for a bigger lineup is rejected, not silently sent
    /// with a stray `$optionD` in the text.
    #[test]
    fn test_validate_lineup_rejects_oversized_template() {
        let template = "$criterion $optionA $optionB $optionC $optionD";
        let result = validate_lineup_template(template, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("$optionD"), "message should name the extra variable: {msg}");
    }

    /// The 4-item default template is valid at 4 and rejected at 3 and 5 —
    /// templates are written for one specific lineup size.
    #[test]
    fn test_lineup_template_is_size_specific() {
        let four = default_lineup_template(4);
        validate_lineup_template(&four, 4).unwrap();
        assert!(validate_lineup_template(&four, 3).is_err());
        assert!(validate_lineup_template(&four, 5).is_err());
    }

    #[test]
    fn test_lineup_item_text_with_tokens_passes_through() {
        let prompt = build_lineup_prompt(
            "$criterion|$optionA|$optionB|$optionC",
            "crit",
            &["has $optionB inside", "plain", "also $length here"],
            "brief",
        );
        assert_eq!(prompt, "crit|has $optionB inside|plain|also $length here");
    }

    #[test]
    fn test_build_prompt_no_reasoning() {
        let prompt = build_prompt(DEFAULT_TEMPLATE_NO_REASONING, "Which is tastier?", "Pizza", "Sushi", "pizza", "sushi", "ignored");
        assert!(prompt.contains("Option 1:\nPizza"));
        assert!(prompt.contains("Option 2:\nSushi"));
        assert!(prompt.contains("Respond only with one of these lines verbatim:"));
        assert!(!prompt.contains("analysis"));
    }
}
