use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use nanojudge_core::{
    constants::{MAX_LINEUP_SIZE, MIN_LINEUP_SIZE},
    run_scoring, judge_hash, winner_dist_to_edges, Edge, JudgeInfo, ScoringOptions,
};

use crate::args::{OutputFormat, ScoreArgs};
use crate::bail;
use crate::output;
use crate::rank::{temper_verdict, temper_verdict_in_place};
use crate::resolve::{DEFAULT_VERDICT_TEMPERATURE_REASONING, DEFAULT_VERDICT_TEMPERATURE_NO_REASONING};
use crate::{
    DEFAULT_BIAS_PRIOR, DEFAULT_BIAS_PRIOR_TAU2, DEFAULT_CONFIDENCE_LEVEL, DEFAULT_PRIOR_TAU2,
    DEFAULT_REGULARIZATION_STRENGTH,
};

pub fn run(args: ScoreArgs) {
    let path = &args.file;
    if !path.exists() {
        bail(format!("File not found: {}", path.display()));
    }

    let bias_prior = args.bias_prior.unwrap_or(DEFAULT_BIAS_PRIOR);
    if !bias_prior.is_finite() || bias_prior <= 0.0 || bias_prior >= 1.0 {
        bail("--bias-prior must be greater than 0.0 and less than 1.0");
    }
    let bias_prior_logit = (bias_prior / (1.0 - bias_prior)).ln();

    let confidence_level = args.confidence_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
    if !confidence_level.is_finite() || confidence_level <= 0.0 || confidence_level >= 1.0 {
        bail(format!("confidence-level={confidence_level}, must be between 0.0 and 1.0 (exclusive)"));
    }
    let regularization_strength = args.regularization_strength.unwrap_or(DEFAULT_REGULARIZATION_STRENGTH);
    if !regularization_strength.is_finite() || regularization_strength <= 0.0 {
        bail(format!("regularization-strength={regularization_strength}, must be finite and > 0"));
    }
    let prior_tau2 = args.prior_tau2.unwrap_or(DEFAULT_PRIOR_TAU2);
    if !prior_tau2.is_finite() || prior_tau2 <= 0.0 {
        bail(format!("prior-tau2={prior_tau2}, must be finite and > 0"));
    }
    let bias_prior_tau2 = args.bias_prior_tau2.unwrap_or(DEFAULT_BIAS_PRIOR_TAU2);
    if !bias_prior_tau2.is_finite() || bias_prior_tau2 <= 0.0 {
        bail(format!("bias-prior-tau2={bias_prior_tau2}, must be finite and > 0"));
    }

    if let Some(t) = args.verdict_temperature {
        if !t.is_finite() || t <= 0.0 {
            bail(format!("verdict-temperature={t}, must be finite and > 0"));
        }
    }

    let mut per_judge_temps: HashMap<String, f64> = HashMap::new();
    for spec in &args.judge_verdict_temperature {
        let eq_pos = spec.rfind('=').unwrap_or_else(|| {
            bail(format!("invalid --judge-verdict-temperature: expected MODEL@ENDPOINT=TEMPERATURE, got: {spec}"))
        });
        let key = &spec[..eq_pos];
        if !key.contains('@') {
            bail(format!("invalid --judge-verdict-temperature: expected MODEL@ENDPOINT=TEMPERATURE, got: {spec}"));
        }
        let value: f64 = spec[eq_pos + 1..].parse().unwrap_or_else(|_| {
            bail(format!("invalid temperature in --judge-verdict-temperature: {spec}"))
        });
        if !value.is_finite() || value <= 0.0 {
            bail(format!("judge-verdict-temperature {key}: {value} must be finite and > 0"));
        }
        if per_judge_temps.contains_key(key) {
            bail(format!("--judge-verdict-temperature {key} specified more than once"));
        }
        per_judge_temps.insert(key.to_string(), value);
    }

    let output_format = args.output_format.unwrap_or_else(|| {
        if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            OutputFormat::Table
        } else {
            OutputFormat::Json
        }
    });

    let (edges, item_names, judge_ids, judge_display_names, judge_flag_keys, total_judgements, logprobs_mode, judge_temps_used, _) =
        load_edges(path, args.verdict_temperature, &per_judge_temps, None);

    if edges.is_empty() {
        bail("No valid judgements found in the file");
    }

    for spec_key in per_judge_temps.keys() {
        if !judge_flag_keys.contains(spec_key) {
            bail(format!(
                "--judge-verdict-temperature {spec_key} does not match any judge in the file (judges found: {})",
                judge_flag_keys.join(", ")
            ));
        }
    }

    let first_temp = judge_temps_used[&judge_ids[0]];
    let all_same = judge_temps_used.values().all(|&t| t == first_temp);
    if all_same {
        eprintln!("Verdict temperature: {first_temp}");
    } else {
        eprintln!("Verdict temperatures:");
        for (i, &jid) in judge_ids.iter().enumerate() {
            eprintln!("  {}: {}", judge_display_names[i], judge_temps_used[&jid]);
        }
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
            scoring_result.panel_positional_bias,
            scoring_result.panel_positional_bias_ci,
        ),
    }
}

fn resolve_edge_temperature(
    judge_model: &str,
    judge_endpoint: &str,
    judge_id: u64,
    per_judge: &HashMap<String, f64>,
    global: Option<f64>,
    judge_reasoning_seen: &mut HashMap<u64, (bool, usize)>,
    record: &serde_json::Value,
    path: &Path,
    line_num: usize,
) -> f64 {
    let judge_key = format!("{judge_model}@{judge_endpoint}");
    if let Some(&t) = per_judge.get(&judge_key) {
        return t;
    }
    if let Some(t) = global {
        return t;
    }
    let reasoning = match record["reasoning"].as_bool() {
        Some(b) => b,
        None => bail(format!(
            "{}:{}: missing reasoning field (required to infer default verdict temperature; pass --verdict-temperature to set explicitly)",
            path.display(), line_num
        )),
    };
    if let Some(&(prev_reasoning, first_line)) = judge_reasoning_seen.get(&judge_id) {
        if reasoning != prev_reasoning {
            bail(format!(
                "{}:{}: judge {judge_model}@{judge_endpoint} has reasoning={reasoning}, but line {first_line} had reasoning={prev_reasoning}",
                path.display(), line_num
            ));
        }
    } else {
        judge_reasoning_seen.insert(judge_id, (reasoning, line_num));
    }
    if reasoning {
        DEFAULT_VERDICT_TEMPERATURE_REASONING
    } else {
        DEFAULT_VERDICT_TEMPERATURE_NO_REASONING
    }
}

/// `expected_lineup_size`: when `Some(k)`, every record must describe a
/// `k`-item judgement (pairwise = 2); a differing size is a hard error. This is
/// how `rank --load-judgements` enforces "loaded file must match the run's
/// lineup size". `None` (the `score` path) accepts any mix of sizes unchanged.
pub(crate) fn load_edges(
    path: &Path,
    global_verdict_temperature: Option<f64>,
    per_judge_verdict_temperatures: &HashMap<String, f64>,
    expected_lineup_size: Option<usize>,
) -> (Vec<Edge>, Vec<String>, Vec<u64>, Vec<String>, Vec<String>, usize, bool, HashMap<u64, f64>, Vec<String>) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| bail(format!("Failed to open {}: {e}", path.display())));
    let reader = std::io::BufReader::new(file);

    let mut item_to_id: HashMap<String, i64> = HashMap::new();
    let mut item_names: Vec<String> = Vec::new();
    let mut item_keys: Vec<String> = Vec::new();
    let mut judge_id_set: Vec<u64> = Vec::new();
    let mut judge_display_names: Vec<String> = Vec::new();
    let mut judge_flag_keys: Vec<String> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut total_judgements: usize = 0;
    let mut logprobs_mode = false;
    let mut judge_reasoning_seen: HashMap<u64, (bool, usize)> = HashMap::new();
    let mut judge_temps_used: HashMap<u64, f64> = HashMap::new();
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
        let judge_id = judge_hash(judge_endpoint, judge_model);

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
            if let Some(expected) = expected_lineup_size
                && items_arr.len() != expected
            {
                bail(format!(
                    "{}:{}: file has a {}-item lineup judgement but this run uses lineup size {}; a loaded file must match the run's lineup size",
                    path.display(), line_num, items_arr.len(), expected
                ));
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

            let hashes: Vec<&str> = match record.get("item_text_hashes") {
                Some(v) => match v.as_array() {
                    Some(arr) if arr.len() == items_arr.len() => {
                        match arr.iter().map(|v| v.as_str()).collect::<Option<Vec<&str>>>() {
                            Some(h) => h,
                            None => bail(format!("{}:{}: non-string item_text_hashes entry", path.display(), line_num)),
                        }
                    }
                    Some(_) => bail(format!("{}:{}: item_text_hashes length mismatch", path.display(), line_num)),
                    None => bail(format!("{}:{}: malformed item_text_hashes", path.display(), line_num)),
                },
                None => bail(format!("{}:{}: missing item_text_hashes", path.display(), line_num)),
            };

            let keys_and_names: Vec<(String, &str)> = items_arr.iter().enumerate().map(|(i, v)| {
                let name = v.as_str()
                    .unwrap_or_else(|| bail(format!("{}:{}: non-string item name", path.display(), line_num)));
                (format!("h:{}", hashes[i]), name)
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
                get_or_insert_item(&mut item_to_id, &mut item_names, &mut item_keys, key, name)
            }).collect();

            let edge_temp = resolve_edge_temperature(
                judge_model, judge_endpoint, judge_id,
                per_judge_verdict_temperatures, global_verdict_temperature,
                &mut judge_reasoning_seen,
                &record, path, line_num,
            );
            judge_temps_used.entry(judge_id).or_insert(edge_temp);

            temper_verdict_in_place(&mut winner_dist, edge_temp);

            if !judge_id_set.contains(&judge_id) {
                judge_id_set.push(judge_id);
                judge_display_names.push(format!("{judge_model} @ {judge_endpoint}"));
                judge_flag_keys.push(format!("{judge_model}@{judge_endpoint}"));
            }
            if line_logprobs {
                logprobs_mode = true;
            }

            let lineup_edges = winner_dist_to_edges(&item_ids, &winner_dist, judge_id, line_logprobs);
            edges.extend(lineup_edges);
            total_judgements += 1;
        } else if record.get("item1").is_some() {
            // Pairwise record
            if let Some(expected) = expected_lineup_size
                && expected != 2
            {
                bail(format!(
                    "{}:{}: file has a pairwise (2-item) judgement but this run uses lineup size {}; a loaded file must match the run's lineup size",
                    path.display(), line_num, expected
                ));
            }
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
                Some(v) => match v.as_str() {
                    Some(h) => format!("h:{h}"),
                    None => bail(format!("{}:{}: malformed item1_text_hash", path.display(), line_num)),
                },
                None => bail(format!("{}:{}: missing item1_text_hash", path.display(), line_num)),
            };
            let key2 = match record.get("item2_text_hash") {
                Some(v) => match v.as_str() {
                    Some(h) => format!("h:{h}"),
                    None => bail(format!("{}:{}: malformed item2_text_hash", path.display(), line_num)),
                },
                None => bail(format!("{}:{}: missing item2_text_hash", path.display(), line_num)),
            };
            if key1 == key2 {
                eprintln!("Warning: {}:{}: pairwise items resolve to same identity, skipping", path.display(), line_num);
                continue;
            }

            let id1 = get_or_insert_item(&mut item_to_id, &mut item_names, &mut item_keys, &key1, item1_name);
            let id2 = get_or_insert_item(&mut item_to_id, &mut item_names, &mut item_keys, &key2, item2_name);

            let edge_temp = resolve_edge_temperature(
                judge_model, judge_endpoint, judge_id,
                per_judge_verdict_temperatures, global_verdict_temperature,
                &mut judge_reasoning_seen,
                &record, path, line_num,
            );
            judge_temps_used.entry(judge_id).or_insert(edge_temp);

            if !judge_id_set.contains(&judge_id) {
                judge_id_set.push(judge_id);
                judge_display_names.push(format!("{judge_model} @ {judge_endpoint}"));
                judge_flag_keys.push(format!("{judge_model}@{judge_endpoint}"));
            }
            if line_logprobs {
                logprobs_mode = true;
            }

            let tempered = temper_verdict(probs, edge_temp);
            edges.push(Edge::new(id1, id2, tempered, judge_id));
            total_judgements += 1;
        } else {
            eprintln!("Warning: {}:{}: unrecognized record format, skipping", path.display(), line_num);
        }
    }

    // Sort items by identity key so the internal index assignment is
    // deterministic regardless of JSONL record order, matching the
    // hash-sorted order used by `rank`.
    let n = item_names.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| item_keys[a].cmp(&item_keys[b]));

    let mut remap = vec![0i64; n];
    let mut sorted_names = Vec::with_capacity(n);
    let mut sorted_hash_keys = Vec::with_capacity(n);
    for (new_id, &old_id) in order.iter().enumerate() {
        remap[old_id] = new_id as i64;
        sorted_names.push(item_names[old_id].clone());
        // Identity keys are always "h:{hash}" (see get_or_insert callers). Hand
        // back the bare hash, in the same order as the returned names/edges, so
        // `rank` can match loaded items against its own text_hashes.
        sorted_hash_keys.push(
            item_keys[old_id]
                .strip_prefix("h:")
                .expect("item identity key is always h:-prefixed")
                .to_string(),
        );
    }
    for edge in &mut edges {
        edge.item1 = remap[edge.item1 as usize];
        edge.item2 = remap[edge.item2 as usize];
    }

    (edges, sorted_names, judge_id_set, judge_display_names, judge_flag_keys, total_judgements, logprobs_mode, judge_temps_used, sorted_hash_keys)
}

fn get_or_insert_item(
    map: &mut HashMap<String, i64>,
    names: &mut Vec<String>,
    keys: &mut Vec<String>,
    key: &str,
    display_name: &str,
) -> i64 {
    if let Some(&id) = map.get(key) {
        id
    } else {
        let id = names.len() as i64;
        map.insert(key.to_string(), id);
        names.push(display_name.to_string());
        keys.push(key.to_string());
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
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"B","item2":"C","item1_text_hash":"b0e6004ac03e61d2","item2_text_hash":"9188835ed6d49e09","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, judges, _, _, total, logprobs, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(edges.len(), 2);
        assert_eq!(names, vec!["A", "C", "B"]);
        assert_eq!(judges.len(), 1);
        assert_eq!(total, 2);
        assert!(logprobs);
        assert!((edges[0].category_probs[0] - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_hash_keys_align_with_names() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"B","item2":"C","item1_text_hash":"b0e6004ac03e61d2","item2_text_hash":"9188835ed6d49e09","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (_, names, _, _, _, _, _, _, hash_keys) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        // Names come back hash-sorted (A, C, B); the hash keys must be in that
        // same order and bare (no "h:" prefix), so an index into one indexes the
        // other. This is what `rank` relies on to remap loaded items.
        assert_eq!(names, vec!["A", "C", "B"]);
        assert_eq!(
            hash_keys,
            vec!["34482beefb0cc992", "9188835ed6d49e09", "b0e6004ac03e61d2"]
        );
    }

    #[test]
    fn test_load_lineup_edges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X","Y","Z"],"item_text_hashes":["1e53ad202eec08bb","aed6adde9f66ae60","efbff0fd345d4a0d"],"winner_dist":[0.5,0.3,0.2],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(names, vec!["X", "Y", "Z"]);
        assert_eq!(total, 1);
        assert!(edges.len() >= 2);
    }

    #[test]
    fn test_load_multiple_judges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m1","judge_endpoint":"http://e1","logprobs":false}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.4,0.6],"judge_model":"m2","judge_endpoint":"http://e2","logprobs":false}"#,
        ]);
        let (edges, _, judges, display_names, _, _, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(edges.len(), 2);
        assert_eq!(judges.len(), 2);
        assert_ne!(judges[0], judges[1]);
        assert!(display_names[0].contains("m1"));
        assert!(display_names[1].contains("m2"));
    }

    #[test]
    fn test_verdict_temperature_applied() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, _, _, _, _, _, _, _, _) = load_edges(f.path(), Some(3.0), &HashMap::new(), None);
        assert!(edges[0].category_probs[0] < 0.9);
        assert!(edges[0].category_probs[0] > 0.5);
    }

    #[test]
    fn test_empty_lines_skipped() {
        let f = write_jsonl(&[
            "",
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":false}"#,
            "",
        ]);
        let (edges, _, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(edges.len(), 1);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_text_hashes_used_for_identity() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"Long title trun...","item2":"B","item1_text_hash":"00000000000000ab","item2_text_hash":"00000000000000cd","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":false}"#,
            r#"{"refit":0,"item1":"Long title trun...","item2":"B","item1_text_hash":"00000000000000ef","item2_text_hash":"00000000000000cd","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":false}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 2);
        assert_eq!(names.len(), 3);
        assert_eq!(edges[0].item1, 0);
        assert_eq!(edges[1].item1, 2);
    }

    #[test]
    fn test_lineup_size_out_of_range_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X"],"winner_dist":[1.0],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_invalid_category_probs_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.0,0.0],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","item1_text_hash":"9188835ed6d49e09","item2_text_hash":"3cec0b494074c4b1","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["D", "C"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_invalid_winner_dist_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["X","Y","Z"],"winner_dist":[0.0,0.0,0.0],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_missing_logprobs_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e"}"#,
            r#"{"refit":0,"item1":"C","item2":"D","item1_text_hash":"9188835ed6d49e09","item2_text_hash":"3cec0b494074c4b1","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["D", "C"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_skipped_record_leaves_no_phantom_items() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["Ghost1","Ghost2","Ghost3"],"winner_dist":[0.5,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_skipped_record_leaves_no_phantom_judges() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.0,0.0],"judge_model":"phantom","judge_endpoint":"http://phantom","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"real","judge_endpoint":"http://real","logprobs":true}"#,
        ]);
        let (_, _, judges, display_names, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(judges.len(), 1);
        assert!(display_names[0].contains("real"));
    }

    #[test]
    fn test_duplicate_lineup_items_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"items":["Same","Same","Other"],"item_text_hashes":["4b40ab569b4eb741","4b40ab569b4eb741","2d12788030f6dda9"],"winner_dist":[0.5,0.3,0.2],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_pairwise_self_edge_skipped() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"A","item1_text_hash":"34482beefb0cc992","item2_text_hash":"34482beefb0cc992","category_probs":[0.7,0.3],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","item1_text_hash":"9188835ed6d49e09","item2_text_hash":"3cec0b494074c4b1","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
        ]);
        let (edges, names, _, _, _, total, _, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
        assert_eq!(total, 1);
        assert_eq!(names, vec!["D", "C"]);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_logprobs_mode_not_set_by_skipped_records() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","category_probs":[0.0,0.0],"judge_model":"m","judge_endpoint":"http://e","logprobs":true}"#,
            r#"{"refit":0,"item1":"C","item2":"D","item1_text_hash":"9188835ed6d49e09","item2_text_hash":"3cec0b494074c4b1","category_probs":[0.6,0.4],"judge_model":"m","judge_endpoint":"http://e","logprobs":false}"#,
        ]);
        let (_, _, _, _, _, _, logprobs, _, _) = load_edges(f.path(), Some(1.0), &HashMap::new(), None);
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

    #[test]
    fn test_auto_verdict_temperature_reasoning_true() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m","judge_endpoint":"http://e","logprobs":true,"reasoning":true}"#,
        ]);
        let (edges, _, _, _, _, _, _, temps, _) = load_edges(f.path(), None, &HashMap::new(), None);
        assert!(temps.values().all(|&t| t == 3.0));
        assert!(edges[0].category_probs[0] < 0.9);
    }

    #[test]
    fn test_auto_verdict_temperature_reasoning_false() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m","judge_endpoint":"http://e","logprobs":true,"reasoning":false}"#,
        ]);
        let (edges, _, _, _, _, _, _, temps, _) = load_edges(f.path(), None, &HashMap::new(), None);
        assert!(temps.values().all(|&t| t == 1.0));
        assert!((edges[0].category_probs[0] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_per_judge_verdict_temperature() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m1","judge_endpoint":"http://e1","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m2","judge_endpoint":"http://e2","logprobs":true}"#,
        ]);
        let mut per_judge = HashMap::new();
        per_judge.insert("m1@http://e1".to_string(), 1.0);
        per_judge.insert("m2@http://e2".to_string(), 3.0);
        let (edges, _, judges, _, _, _, _, temps, _) = load_edges(f.path(), None, &per_judge, None);
        assert_eq!(judges.len(), 2);
        let j1 = judge_hash("http://e1", "m1");
        let j2 = judge_hash("http://e2", "m2");
        assert_eq!(temps[&j1], 1.0);
        assert_eq!(temps[&j2], 3.0);
        let e1 = edges.iter().find(|e| e.judge_id == j1).unwrap();
        let e2 = edges.iter().find(|e| e.judge_id == j2).unwrap();
        assert!((e1.category_probs[0] - 0.9).abs() < 1e-9);
        assert!(e2.category_probs[0] < 0.9);
    }

    #[test]
    fn test_per_judge_overrides_global() {
        let f = write_jsonl(&[
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m1","judge_endpoint":"http://e1","logprobs":true}"#,
            r#"{"refit":0,"item1":"A","item2":"B","item1_text_hash":"34482beefb0cc992","item2_text_hash":"b0e6004ac03e61d2","category_probs":[0.9,0.1],"judge_model":"m2","judge_endpoint":"http://e2","logprobs":true}"#,
        ]);
        let mut per_judge = HashMap::new();
        per_judge.insert("m1@http://e1".to_string(), 1.0);
        let (_, _, _, _, _, _, _, temps, _) = load_edges(f.path(), Some(3.0), &per_judge, None);
        let j1 = judge_hash("http://e1", "m1");
        let j2 = judge_hash("http://e2", "m2");
        assert_eq!(temps[&j1], 1.0);
        assert_eq!(temps[&j2], 3.0);
    }
}
