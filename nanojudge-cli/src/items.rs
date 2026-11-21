use std::io::{self, BufRead, IsTerminal};
use std::path::Path;

use crate::args::RankArgs;
use crate::bail;

const TITLE_MAX_LEN: usize = 20;

/// Plain text file extensions that we read from directories.
const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "html", "csv", "json", "xml", "rst", "log", "yaml", "yml", "toml",
];

/// Derive a display title from item text: up to 20 chars, preferring a word boundary.
fn title_from_text(text: &str) -> String {
    if text.chars().count() <= TITLE_MAX_LEN {
        return text.to_string();
    }
    let truncated: String = text.chars().take(TITLE_MAX_LEN).collect();
    match truncated.rfind(' ') {
        Some(pos) => format!("{}...", &truncated[..pos]),
        None => format!("{truncated}..."),
    }
}

/// Parse a string as either a JSON array of strings or plain text (one item per line).
fn parse_items_from_str(content: &str) -> Result<Vec<String>, String> {
    let trimmed = content.trim();
    if trimmed.starts_with('[') {
        // Try JSON array
        let items: Vec<String> = serde_json::from_str(trimmed)
            .map_err(|e| format!("File looks like JSON but failed to parse: {e}"))?;
        Ok(items.into_iter().filter(|s| !s.trim().is_empty()).collect())
    } else {
        // Plain text, one item per line
        Ok(trimmed
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }
}

/// Load items from all sources: --items file/dir, --item inline args, or stdin.
/// Returns (titles, texts) where titles are for display and texts are sent to the LLM.
pub fn load_items(args: &RankArgs) -> (Vec<String>, Vec<String>) {
    load_items_from(args.items.as_deref(), &args.inline_items, || {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            return Err(
                "No items provided. Use --items <file|dir>, --item <name>, or pipe items via stdin.".into(),
            );
        }
        stdin
            .lock()
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .map(|lines| lines.join("\n"))
            .map_err(|e| format!("Failed to read from stdin: {e}"))
    })
    .unwrap_or_else(|msg| bail(msg))
}

fn load_items_from(
    items_path: Option<&Path>,
    inline_items: &[String],
    read_stdin: impl FnOnce() -> Result<String, String>,
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut titles = Vec::new();
    let mut texts = Vec::new();

    if let Some(path) = items_path {
        if path.is_dir() {
            // Directory mode: each file is an item
            let entries = std::fs::read_dir(path)
                .map_err(|e| format!("Failed to read directory {}: {e}", path.display()))?;

            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .collect();

            // Sort by filename for deterministic ordering
            files.sort_by_key(|e| e.file_name());

            let mut skipped = 0usize;
            let total = files.len();

            for entry in &files {
                let file_path = entry.path();
                let ext = file_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);

                if !ext.as_deref().is_some_and(|e| TEXT_EXTENSIONS.contains(&e)) {
                    skipped += 1;
                    continue;
                }

                let content = std::fs::read_to_string(&file_path)
                    .map_err(|e| format!("Failed to read {}: {e}", file_path.display()))?;
                let content = content.trim().to_string();

                if content.is_empty() {
                    skipped += 1;
                    continue;
                }

                let stem = file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unnamed")
                    .to_string();

                titles.push(stem);
                texts.push(content);
            }

            let loaded = titles.len();
            eprintln!("Found {total} files, loaded {loaded}, skipped {skipped} (unsupported format or empty)");
        } else {
            // File mode: one item per line or JSON array
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read items file {}: {e}", path.display()))?;
            texts = parse_items_from_str(&content)?;
            titles = texts.iter().map(|t| title_from_text(t)).collect();
        }
    }

    // From inline --item flags
    for item in inline_items {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            return Err("Inline --item cannot be empty".into());
        }
        titles.push(title_from_text(trimmed));
        texts.push(trimmed.to_string());
    }

    // From stdin (only if no file/dir and no inline items)
    if items_path.is_none() && inline_items.is_empty() {
        let content = read_stdin()?;
        texts = parse_items_from_str(&content)?;
        titles = texts.iter().map(|t| title_from_text(t)).collect();
    }

    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, text) in texts.iter().enumerate() {
        if let Some(&first) = seen.get(text.as_str()) {
            return Err(format!(
                "Duplicate item text: \"{}\" appears at positions {} and {} (1-indexed)",
                title_from_text(text),
                first + 1,
                i + 1,
            ));
        }
        seen.insert(text, i);
    }

    if texts.len() < 2 {
        return Err(format!("Need at least 2 items to rank, got {}", texts.len()));
    }
    Ok((titles, texts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unused_stdin() -> Result<String, String> {
        panic!("stdin should not be read when --items or --item is provided");
    }

    #[test]
    fn empty_items_file_does_not_fall_through_to_stdin() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let err = load_items_from(Some(tmp.path()), &[], unused_stdin).unwrap_err();
        assert!(err.contains("Need at least 2 items to rank, got 0"), "{err}");
    }

    #[test]
    fn directory_loads_uppercase_extensions_case_insensitively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("A.TXT"), "first item").unwrap();
        std::fs::write(dir.path().join("B.MD"), "second item").unwrap();
        std::fs::write(dir.path().join("C.PNG"), "not text").unwrap();

        let (titles, texts) = load_items_from(Some(dir.path()), &[], unused_stdin).unwrap();
        assert_eq!(titles, vec!["A", "B"]);
        assert_eq!(texts, vec!["first item", "second item"]);
    }

    #[test]
    fn inline_items_are_trimmed_and_blank_values_rejected() {
        let items = vec!["  alpha  ".into(), "beta".into()];
        let (titles, texts) = load_items_from(None, &items, unused_stdin).unwrap();
        assert_eq!(titles, vec!["alpha", "beta"]);
        assert_eq!(texts, vec!["alpha", "beta"]);

        for blank in ["", "   ", "\t\n"] {
            let err = load_items_from(None, &["alpha".into(), blank.into()], unused_stdin).unwrap_err();
            assert!(err.contains("cannot be empty"), "blank {blank:?}: {err}");
        }
    }

    #[test]
    fn stdin_is_used_when_no_items_or_inline_args_given() {
        let (titles, texts) = load_items_from(None, &[], || Ok("first\nsecond\n".into())).unwrap();
        assert_eq!(titles, vec!["first", "second"]);
        assert_eq!(texts, vec!["first", "second"]);

        // Also verify file mode with JSON array works through load_items_from.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, r#"["one", "two"]"#).unwrap();
        let (_, texts) = load_items_from(Some(tmp.path()), &[], unused_stdin).unwrap();
        assert_eq!(texts, vec!["one", "two"]);
    }
}
