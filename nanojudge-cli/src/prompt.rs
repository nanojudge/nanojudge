/// Prompt building for pairwise comparisons.
///
/// Supports custom prompt templates with variable substitution.
/// Required variables: $criterion, $option1, $option2, $length
///
/// If no template is provided, a sensible default is used that produces
/// a "Verdict:" marker followed by a likert scale letter (A-E).

use crate::bail;

pub const DEFAULT_TEMPLATE: &str = "\
$criterion

Option 1:
$option1

Option 2:
$option2

Instructions:
Write a $length analysis. You MUST end your response with one of these sentences verbatim:

Verdict A: Option 1 clearly wins
Verdict B: Option 1 narrowly wins
Verdict C: Draw
Verdict D: Option 2 narrowly wins
Verdict E: Option 2 clearly wins
";

const REQUIRED_VARIABLES: &[&str] = &["$criterion", "$option1", "$option2", "$length"];

/// Validate that a template contains all required variables.
/// Returns an error message listing any missing variables.
pub fn validate_template(template: &str) -> Result<(), String> {
    let missing: Vec<&&str> = REQUIRED_VARIABLES
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
pub fn load_template(path: &std::path::Path) -> String {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| bail(format!("Failed to read prompt template {}: {e}", path.display())));

    if let Err(msg) = validate_template(&content) {
        bail(format!("{} (in {})", msg, path.display()));
    }

    content
}

/// Build a comparison prompt by substituting variables into a template.
pub fn build_prompt(template: &str, criterion: &str, option1: &str, option2: &str, analysis_length: &str) -> String {
    // Trim trailing "s" from length descriptor for grammar ("3-5 paragraph" not "3-5 paragraphs")
    let length = analysis_length.trim_end_matches('s');
    template
        .replace("$criterion", criterion)
        .replace("$option1", option1)
        .replace("$option2", option2)
        .replace("$length", length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_template_is_valid() {
        validate_template(DEFAULT_TEMPLATE).unwrap();
    }

    #[test]
    fn test_build_prompt_with_default_template() {
        let prompt = build_prompt(DEFAULT_TEMPLATE, "Which is tastier?", "Pizza", "Sushi", "2 paragraphs");
        assert!(prompt.starts_with("Which is tastier?"));
        assert!(prompt.contains("Option 1:\nPizza"));
        assert!(prompt.contains("Option 2:\nSushi"));
        assert!(prompt.contains("2 paragraph"));
        assert!(prompt.contains("Verdict A:"));
        assert!(prompt.contains("Verdict A: Option 1 clearly wins"));
        assert!(prompt.contains("E: Option 2 clearly wins"));
    }

    #[test]
    fn test_custom_template() {
        let template = "Compare $option1 vs $option2 for $criterion. Be $length.";
        let prompt = build_prompt(template, "taste", "Pizza", "Sushi", "brief");
        assert_eq!(prompt, "Compare Pizza vs Sushi for taste. Be brief.");
    }

    #[test]
    fn test_validate_missing_variables() {
        let result = validate_template("Just $option1 and $option2");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("$criterion"));
        assert!(msg.contains("$length"));
    }

    #[test]
    fn test_validate_complete_template() {
        let template = "$criterion\n$option1\n$option2\n$length";
        validate_template(template).unwrap();
    }
}
