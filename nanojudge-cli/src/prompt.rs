//! Prompt building for pairwise comparisons.
//!
//! Supports custom prompt templates with variable substitution.
//! Required variables: $criterion, $option1, $option2
//!
//! If no template is provided, a sensible default is used that produces
//! a "Verdict:" marker followed by a verdict letter (A-D).

use crate::bail;

pub const DEFAULT_TEMPLATE: &str = "\
$criterion

Option 1:
$option1

Option 2:
$option2

Instructions:
Write a $length analysis. You MUST end your response with one of these lines verbatim:

Verdict A: Option 1, clearly
Verdict B: Option 1, marginally
Verdict C: Option 2, marginally
Verdict D: Option 2, clearly
";

pub const DEFAULT_TEMPLATE_NO_REASONING: &str = "\
$criterion

Option 1:
$option1

Option 2:
$option2

Instructions:
Respond only with one of these lines verbatim:

Verdict A: Option 1, clearly
Verdict B: Option 1, marginally
Verdict C: Option 2, marginally
Verdict D: Option 2, clearly
";

const REQUIRED_VARIABLES: &[&str] = &["$criterion", "$option1", "$option2"];
const REQUIRED_VARIABLES_NO_REASONING: &[&str] = &["$criterion", "$option1", "$option2"];

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

/// Load a prompt template from a file path, validate it, and return the contents.
pub fn load_template(path: &std::path::Path, reasoning_enabled: bool) -> String {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| bail(format!("Failed to read prompt template {}: {e}", path.display())));

    if let Err(msg) = validate_template(&content, reasoning_enabled) {
        bail(format!("{} (in {})", msg, path.display()));
    }

    content
}

/// Build a comparison prompt by substituting variables into a template.
///
/// Single-pass substitution: only variables in the TEMPLATE are replaced.
/// Substituted values are never rescanned, so item text containing literal
/// `$option2`, `$length`, etc. (common when ranking code) passes through
/// untouched instead of being recursively substituted.
pub fn build_prompt(template: &str, criterion: &str, option1: &str, option2: &str, name1: &str, name2: &str, analysis_length: &str) -> String {
    // Trim trailing "s" from length descriptor for grammar ("3-5 paragraph" not "3-5 paragraphs")
    let length = analysis_length.trim_end_matches('s');
    let vars: [(&str, &str); 6] = [
        ("$criterion", criterion),
        ("$option1", option1),
        ("$option2", option2),
        ("$name1", name1),
        ("$name2", name2),
        ("$length", length),
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

#[cfg(test)]
mod tests {
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
        assert!(prompt.contains("2 paragraph"));
        assert!(prompt.contains("Verdict A:"));
        assert!(prompt.contains("Verdict A: Option 1, clearly"));
        assert!(prompt.contains("D: Option 2, clearly"));
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
        assert!(DEFAULT_TEMPLATE_NO_REASONING.contains("Verdict A: Option 1, clearly"));
        assert!(DEFAULT_TEMPLATE_NO_REASONING.contains("Verdict D: Option 2, clearly"));
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
