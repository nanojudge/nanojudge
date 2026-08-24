use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use nanojudge_core::{
    constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE},
    run_scoring, stable_hash, winner_dist_to_edges, Edge, JudgeInfo, ScoringOptions,
};

use crate::args::{OutputFormat, ScoreArgs};
use crate::bail;
use crate::output;
use crate::rank::{temper_verdict, temper_verdict_in_place};

pub fn run(args: ScoreArgs) {
    let path = &args.file;
    if !path.exists() {
        bail(format!("File not found: {}", path.display()));
    }

    let bias_prior = args.bias_prior.unwrap_or(0.5);
    if !bias_prior.is_finite() || bias_prior <= 0.0 || bias_prior >= 1.0 {
        bail("--bias-prior must be greater than 0.0 and less than 1.0");
    }
    let bias_prior_logit = (bias_prior / (1.0 - bias_prior)).ln();

    let confidence_level = args.confidence_level.unwrap_or(0.95);
    if !confidence_level.is_finite() || confidence_level <= 0.0 || confidence_level >= 1.0 {
        bail(format!("confidence-level={confidence_level}, must be between 0.0 and 1.0 (exclusive)"));
    }
    let regularization_strength = args.regularization_strength.unwrap_or(0.01);
    if !regularization_strength.is_finite() || regularization_strength <= 0.0 {
        bail(format!("regularization-strength={regularization_strength}, must be finite and > 0"));
    }
    let prior_tau2 = args.prior_tau2.unwrap_or(10.0);
    if !prior_tau2.is_finite() || prior_tau2 <= 0.0 {
        bail(format!("prior-tau2={prior_tau2}, must be finite and > 0"));
    }
    let bias_prior_tau2 = args.bias_prior_tau2.unwrap_or(2.0);
    if !bias_prior_tau2.is_finite() || bias_prior_tau2 <= 0.0 {
        bail(format!("bias-prior-tau2={bias_prior_tau2}, must be finite and > 0"));
    }

    let output_format = args.output_format.unwrap_or_else(|| {
        if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            OutputFormat::Table
        } else {
            OutputFormat::Json
        }
    });

    let (edges, item_names, judge_ids, judge_display_names, total_judgements, logprobs_mode) =
        load_edges(path);

    if edges.is_empty() {
        bail("No valid judgements found in the file");
    }

    let item_ids: Vec<i64> = (0..item_names.len() as i64).collect();

    let judge_info = JudgeInfo {
        judge_ids: judge_ids.clone(),
        logprobs_mode,
    };

    let scoring_result = run_scoring(
        &item_ids,
        &edges,
        &ScoringOptions {
            confidence_level,
            selection_sharpness: None,
            anchor_index: 0.0,
            selection_cutoff: 0.0,
            selection_coverage: 0.0,
            target_prior_edges: 0.0,
            regularization_strength,
            prior_tau2,
            bias_prior_tau2,
            bias_prior_logit,
        },
        &judge_info,
    );

    let mut edge_counts = vec![0usize; item_names.len()];
    for e in &edges {
        edge_counts[e.item1 as usize] += 1;
        edge_counts[e.item2 as usize] += 1;
    }

    let judge_name_map: HashMap<u64, String> = judge_ids.iter()
        .zip(judge_display_names.iter())
        .map(|(&id, name)| (id, name.clone()))
        .collect();
    let empty_tokens: HashMap<u64, (u64, u64)> = HashMap::new();
    let empty_wall_time: HashMap<u64, f64> = HashMap::new();

    match output_format {
        OutputFormat::Json => output::print_json(
            &scoring_result.rankings,
            &item_names,
            &edge_counts,
            total_judgements,
            &scoring_result.judge_analytics,
            scoring_result.panel_positional_bias,
            scoring_result.panel_positional_bias_ci,
            None,
        ),
        OutputFormat::Table => output::print_table(
            &scoring_result.rankings,
            &item_names,
            &edge_counts,
            total_judgements,
            confidence_level,
            &scoring_result.judge_analytics,
            &judge_name_map,
            &empty_tokens,
            &empty_wall_time,
        ),
    }
}

fn load_edges(
    path: &Path,
) -> (Vec<Edge>, Vec<String>, Vec<u64>, Vec<String>, usize, bool) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));
    let reader = std::io::BufReader::new(file);

    let mut item_to_id: HashMap<String, i64> = HashMap::new();
    let mut item_names: Vec<String> = Vec::new();
    let mut judge_id_set: Vec<u64> = Vec::new();
    let mut judge_display_names: Vec<String> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut total_judgements: usize = 0;
    let mut logprobs_mode = false;
    let mut line_num: usize = 0;

    for line_result in reader.lines() {
        line_num += 1;
        let line = line_result
            .unwrap_or_else(|e| bail(format!("{}:{}: read error: {e}", path.display(), line_num)));
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let record: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| bail(format!("{}:{}: invalid JSON: {e}", path.display(), line_num)));

        let judge_model = record["judge_model"].as_str()
            .unwrap_or_else(|| bail(format!("{}:{}: missing judge_model", path.display(), line_num)));
        let judge_endpoint = record["judge_endpoint"].as_str()
            .unwrap_or_else(|| bail(format!("{}:{}: missing judge_endpoint", path.display(), line_num)));
        let judge_id = stable_hash(&format!("{judge_endpoint}\0{judge_model}"));

        let verdict_temperature = match record["verdict_temperature"].as_f64() {
            Some(t) if t.is_finite() && t > 0.0 => t,
            Some(t) => {
                eprintln!("Warning: {}:{}: invalid verdict_temperature {t}, skipping", path.display(), line_num);
                continue;
            }
            None => {
                eprintln!("Warning: {}:{}: missing verdict_temperature, skipping", path.display(), line_num);
                continue;
            }
        };
        let line_logprobs = match record["logprobs"].as_bool() {
            Some(b) => b,
            None => {
                eprintln!("Warning: {}:{}: missing logprobs, skipping", path.display(), line_num);
                continue;
            }
        };

        if let Some(items_arr) = record["items"].as_array() {
            // Lineup record
            let winner_dist_arr = match record["winner_dist"].as_array() {
                Some(arr) => arr,
                None => {
                    eprintln!("Warning: {}:{}: lineup record missing winner_dist, skipping", path.display(), line_num);
                    continue;
                }
            };

            if items_arr.len() != winner_dist_arr.len() {
                eprintln!("Warning: {}:{}: items/winner_dist length mismatch, skipping", path.display(), line_num);
                continue;
            }
            if !(MIN_LINEUP_SIZE..=MAX_LINEUP_SIZE).contains(&items_arr.len()) {
                eprintln!("Warning: {}:{}: lineup size {} out of range ({}..={}), skipping", path.display(), line_num, items_arr.len(), MIN_LINEUP_SIZE, MAX_LINEUP_SIZE);
                continue;
            }

            let mut winner_dist: Vec<f64> = winner_dist_arr.iter().map(|v| {
                v.as_f64()
                    .unwrap_or_else(|| bail(format!("{}:{}: non-numeric winner_dist entry", path.display(), line_num)))
            }).collect();

            let wd_sum: f64 = winner_dist.iter().sum();
            if !wd_sum.is_finite() || wd_sum <= 0.0 || winner_dist.iter().any(|&p| p < 0.0) {
                eprintln!("Warning: {}:{}: invalid winner_dist, skipping", path.display(), line_num);
                continue;
            }

            let valid_hashes: Option<Vec<u64>> = record["item_text_hashes"].as_array()
                .and_then(|arr| {
                    if arr.len() != items_arr.len() { return None; }
                    arr.iter().map(|v| v.as_u64()).collect()
                });
            if record.get("item_text_hashes").is_some() && valid_hashes.is_none() {
                eprintln!("Warning: {}:{}: malformed item_text_hashes, skipping", path.display(), line_num);
                continue;
            }

            let keys_and_names: Vec<(String, &str)> = items_arr.iter().enumerate().map(|(i, v)| {
                let name = v.as_str()
                    .unwrap_or_else(|| bail(format!("{}:{}: non-string item name", path.display(), line_num)));
                let key = valid_hashes.as_ref()
                    .map(|hashes| format!("h:{}", hashes[i]))
                    .unwrap_or_else(|| format!("t:{name}"));
                (key, name)
            }).collect();

            let mut has_dup = false;
            for (i, (a, _)) in keys_and_names.iter().enumerate() {
                for (b, _) in &keys_and_names[i + 1..] {
                    if a == b { has_dup = true; break; }
                }
                if has_dup { break; }
            }
            if has_dup {
                eprintln!("Warning: {}:{}: lineup contains duplicate items after identity resolution, skipping", path.display(), line_num);
                continue;
            }

            let item_ids: Vec<i64> = keys_and_names.iter().map(|(key, name)| {
                get_or_insert_item(&mut item_to_id, &mut item_names, key, name)
            }).collect();

            temper_verdict_in_place(&mut winner_dist, verdict_temperature);

            if !judge_id_set.contains(&judge_id) {
                judge_id_set.push(judge_id);
                judge_display_names.push(format!("{judge_model} @ {judge_endpoint}"));
            }
            if line_logprobs {
                logprobs_mode = true;
            }

            let lineup_edges = winner_dist_to_edges(&item_ids, &winner_dist, judge_id, line_logprobs);
            edges.extend(lineup_edges);
            total_judgements += 1;
        } else if record.get("item1").is_some() {
            // Pairwise record
            let item1_name = record["item1"].as_str()
                .unwrap_or_else(|| bail(format!("{}:{}: missing item1", path.display(), line_num)));
            let item2_name = record["item2"].as_str()
                .unwrap_or_else(|| bail(format!("{}:{}: missing item2", path.display(), line_num)));

            let category_probs_arr = match record["category_probs"].as_array() {
                Some(arr) => arr,
                None => {
                    eprintln!("Warning: {}:{}: pairwise record missing category_probs, skipping", path.display(), line_num);
                    continue;
                }
            };
            if category_probs_arr.len() != 2 {
                eprintln!("Warning: {}:{}: category_probs must have exactly 2 elements, skipping", path.display(), line_num);
                continue;
            }

            let probs: [f64; 2] = [
                category_probs_arr[0].as_f64()
                    .unwrap_or_else(|| bail(format!("{}:{}: non-numeric category_probs", path.display(), line_num))),
                category_probs_arr[1].as_f64()
                    .unwrap_or_else(|| bail(format!("{}:{}: non-numeric category_probs", path.display(), line_num))),
            ];

            let sum = probs[0] + probs[1];
            if !sum.is_finite() || sum <= 0.0 || probs[0] < 0.0 || probs[1] < 0.0 {
                eprintln!("Warning: {}:{}: invalid category_probs {:?}, skipping", path.display(), line_num, probs);
                continue;
            }

            let key1 = match record.get("item1_text_hash") {
                Some(v) => match v.as_u64() {
                    Some(h) => format!("h:{h}"),
                    None => {
                        eprintln!("Warning: {}:{}: malformed item1_text_hash, skipping", path.display(), line_num);
                        continue;
                    }
                },
                None => format!("t:{item1_name}"),
            };
            let key2 = match record.get("item2_text_hash") {
                Some(v) => match v.as_u64() {
                    Some(h) => format!("h:{h}"),
                    None => {
                        eprintln!("Warning: {}:{}: malformed item2_text_hash, skipping", path.display(), line_num);
                        continue;
                    }
                },
                None => format!("t:{item2_name}"),
            };
            if key1 == key2 {
                eprintln!("Warning: {}:{}: pairwise items resolve to same identity, skipping", path.display(), line_num);
                continue;
            }

            let id1 = get_or_insert_item(&mut item_to_id, &mut item_names, &key1, item1_name);
            let id2 = get_or_insert_item(&mut item_to_id, &mut item_names, &key2, item2_name);

            if !judge_id_set.contains(&judge_id) {
                judge_id_set.push(judge_id);
                judge_display_names.push(format!("{judge_model} @ {judge_endpoint}"));
            }
            if line_logprobs {
                logprobs_mode = true;
            }

            let tempered = temper_verdict(probs, verdict_temperature);
            edges.push(Edge::new(id1, id2, tempered, judge_id));
            total_judgements += 1;
        } else {
            eprintln!("Warning: {}:{}: unrecognized record format, skipping", path.display(), line_num);
        }
    }

    (edges, item_names, judge_id_set, judge_display_names, total_judgements, logprobs_mode)
}

fn get_or_insert_item(
    map: &mut HashMap<String, i64>,
    names: &mut Vec<String>,
    key: &str,
    display_name: &str,
) -> i64 {
    if let Some(&id) = map.get(key) {
        id
    } else {
        let id = names.len() as i64;
        map.insert(key.to_string(), id);
        names.push(display_name.to_string());
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn test_load_pairwise_edges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"B","item2":"C","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, judges, _, total, logprobs) = load_edges(f.path());
        assert_eq!(edges.len(), 2);
        assert_eq!(names, vec!["A", "B", "C"]);
        assert_eq!(judges.len(), 1);
        assert_eq!(total, 2);
        assert!(logprobs);
        assert!((edges[0].category_probs[0] - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_load_lineup_edges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X","Y","Z"],"winner_dist":[0.5,0.3,0.2],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(names, vec!["X", "Y", "Z"]);
        assert_eq!(total, 1);
        assert!(edges.len() >= 2);
    }

    #[test]
    fn test_load_multiple_judges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m1","judge_endpoint":"http://e1","verdict_temperature":1.0,"logprobs":false}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.4,0.6],"judge_model":"m2","judge_endpoint":"http://e2","verdict_temperature":1.0,"logprobs":false}"#,
        ]);
        let (edges, _, judges, display_names, _, _) = load_edges(f.path());
        assert_eq!(edges.len(), 2);
        assert_eq!(judges.len(), 2);
        assert_ne!(judges[0], judges[1]);
        assert!(display_names[0].contains("m1"));
        assert!(display_names[1].contains("m2"));
    }

    #[test]
    fn test_verdict_temperature_applied() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.9,0.1],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":3.0,"logprobs":true}"#,
        ]);
        let (edges, _, _, _, _, _) = load_edges(f.path());
        assert!(edges[0].category_probs[0] < 0.9);
        assert!(edges[0].category_probs[0] > 0.5);
    }

    #[test]
    fn test_empty_lines_skipped() {
        let f = write_jsonl(&[
            "",
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":false}"#,
            "",
        ]);
        let (edges, _, _, _, total, _) = load_edges(f.path());
        assert_eq!(edges.len(), 1);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_text_hashes_used_for_identity() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"Long title trun...","item2":"B","item1_text_hash":111,"item2_text_hash":222,"category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":false}"#,
            r#"{"refit":0,"item1":"Long title trun...","item2":"B","item1_text_hash":333,"item2_text_hash":222,"category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":false}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 2);
        assert_eq!(names.len(), 3);
        assert_eq!(edges[0].item1, 0);
        assert_eq!(edges[1].item1, 2);
    }

    #[test]
    fn test_lineup_size_out_of_range_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X"],"winner_dist":[1.0],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_invalid_category_probs_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.0,0.0],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["C", "D"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_invalid_winner_dist_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X","Y","Z"],"winner_dist":[0.0,0.0,0.0],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_missing_verdict_temperature_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["C", "D"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_missing_logprobs_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0}"#,
            r#"{"refit":0,"item1":"C","item2":"D","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["C", "D"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_skipped_record_leaves_no_phantom_items() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["Ghost1","Ghost2","Ghost3"],"winner_dist":[0.5,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_skipped_record_leaves_no_phantom_judges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.0,0.0],"judge_model":"phantom","judge_endpoint":"http://phantom","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"real","judge_endpoint":"http://real","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (_, _, judges, display_names, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(judges.len(), 1);
        assert!(display_names[0].contains("real"));
    }

    #[test]
    fn test_malformed_lineup_hashes_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X","Y","Z"],"item_text_hashes":[111,null,333],"winner_dist":[0.5,0.3,0.2],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_duplicate_lineup_items_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["Same","Same","Other"],"item_text_hashes":[100,100,200],"winner_dist":[0.5,0.3,0.2],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_pairwise_self_edge_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"A","item1_text_hash":100,"item2_text_hash":100,"category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["C", "D"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_malformed_pairwise_hash_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"not_a_number","item2_text_hash":222,"category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
        ]);
        let (edges, names, _, _, total, _) = load_edges(f.path());
        assert_eq!(total, 1);
        assert_eq!(names, vec!["C", "D"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_logprobs_mode_not_set_by_skipped_records() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.0,0.0],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","verdict_temperature":1.0,"logprobs":false}"#,
        ]);
        let (_, _, _, _, _, logprobs) = load_edges(f.path());
        assert!(!logprobs);
    }

    #[test]
    fn test_temper_verdict_identity() {
        let result = temper_verdict([0.7, 0.3], 1.0);
        assert!((result[0] - 0.7).abs() < 1e-9);
        assert!((result[1] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_temper_verdict_softens() {
        let result = temper_verdict([0.9, 0.1], 3.0);
        assert!(result[0] < 0.9);
        assert!(result[0] > 0.5);
        assert!((result[0] + result[1] - 1.0).abs() < 1e-9);
    }
}
