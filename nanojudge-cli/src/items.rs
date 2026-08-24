use std::io::{self, BufRead, IsTerminal};

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
fn parse_items_from_str(content: &str) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.starts_with('[') {
        // Try JSON array
        let items: Vec<String> = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| bail(format!("File looks like JSON but failed to parse: {e}")));
        items.into_iter().filter(|s| !s.trim().is_empty()).collect()
    } else {
        // Plain text, one item per line
        trimmed
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// Load items from all sources: --items file/dir, --item inline args, or stdin.
/// Returns (titles, texts) where titles are for display and texts are sent to the LLM.
pub fn load_items(args: &RankArgs) -> (Vec<String>, Vec<String>) {
    let mut titles = Vec::new();
    let mut texts = Vec::new();

    if let Some(ref path) = args.items {
        if path.is_dir() {
            // Directory mode: each file is an item
            let entries = std::fs::read_dir(path)
                .unwrap_or_else(|e| bail(format!("Failed to read directory {}: {e}", path.display())));

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
                let ext = file_path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");

                if !TEXT_EXTENSIONS.contains(&ext) {
                    skipped += 1;
                    continue;
                }

                let content = std::fs::read_to_string(&file_path)
                    .unwrap_or_else(|e| bail(format!("Failed to read {}: {e}", file_path.display())));
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
                .unwrap_or_else(|e| bail(format!("Failed to read items file {}: {e}", path.display())));
            texts = parse_items_from_str(&content);
            titles = texts.iter().map(|t| title_from_text(t)).collect();
        }
    }

    // From inline --item flags
    for item in &args.inline_items {
        titles.push(title_from_text(item));
        texts.push(item.clone());
    }

    // From stdin (only if no file/dir and no inline items)
    if texts.is_empty() {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            bail("No items provided. Use --items <file|dir>, --item <name>, or pipe items via stdin.");
        }
        let content: String = stdin.lock().lines()
            .map(|l| l.expect("Failed to read from stdin"))
            .collect::<Vec<_>>()
            .join("\n");
        texts = parse_items_from_str(&content);
        titles = texts.iter().map(|t| title_from_text(t)).collect();
    }

    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, text) in texts.iter().enumerate() {
        if let Some(&first) = seen.get(text.as_str()) {
            bail(format!(
                "Duplicate item text: \"{}\" appears at positions {} and {} (1-indexed)",
                title_from_text(text),
                first + 1,
                i + 1,
            ));
        }
        seen.insert(text, i);
    }

    if texts.len() < 2 {
        bail(format!("Need at least 2 items to rank, got {}", texts.len()));
    }
    (titles, texts)
}
