//! Voyag V8 local inference for Rust.
//!
//! Reproduces the exact pipeline of the Python runner `inference.py`:
//! schema + tokenizer (Metaspace BPE via the `tokenizers` crate) + fixed
//! [10, 512] ONNX input + decode.
//!
//! ```no_run
//! let predictor = voyag::VoyagPredictor::create("path/to/voyag_github_release")?;
//! let result = predictor.predict("打开车窗，同时把空调调到24度", None)?;
//! println!("{result}");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::error::Error;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

pub const MODEL_NAME: &str = "voyag-v8-onnx";

const CANONICAL_STATE_FIELDS: [&str; 15] = [
    "utterance", "speed_kmh", "acc", "window_state", "hvac_state",
    "hvac_temperature_c", "media_state", "seat_heat", "light_state",
    "phone_call", "nav_active", "battery_soc", "gps_valid", "dtc", "warning",
];

/// Loaded predictor: schema + tokenizer + ONNX session, kept for repeated calls.
pub struct VoyagPredictor {
    schema: Value,
    tokenizer: Tokenizer,
    session: Session,
    slot_ids: Vec<String>,
    relation_labels: Vec<String>,
    max_len: usize,
    head_max_len: usize,
    /// tokenizer.json `truncation.max_length`; Python applies it on every encode.
    truncation_max: usize,
    pad_id: i64,
    cls_id: i64,
    sep_id: i64,
    mask_id: i64,
    mask_token: String,
    pairs: Vec<(usize, usize)>,
    /// Pre-encoded slot-question heads: identical for every request.
    head_cache: Vec<(Vec<i64>, Vec<i64>)>,
}

fn neutral_state() -> Value {
    serde_json::json!({
        "utterance": "", "speed_kmh": 0, "acc": "off",
        "window_state": "closed", "hvac_state": "off",
        "hvac_temperature_c": 22, "media_state": "idle",
        "seat_heat": "off", "light_state": "off",
        "phone_call": "idle", "nav_active": false,
        "battery_soc": 0.70, "gps_valid": true,
        "dtc": [], "warning": [],
    })
}

fn canonicalize_state(state: &Value) -> Result<Value, Box<dyn Error>> {
    let obj = state.as_object().ok_or("state must be a JSON object")?;
    if !obj.contains_key("utterance") {
        return Err("state must contain 'utterance'".into());
    }
    let mut out = Map::new();
    for key in CANONICAL_STATE_FIELDS {
        if let Some(value) = obj.get(key) {
            out.insert(key.to_string(), value.clone());
        }
    }
    for (key, value) in obj {
        if !out.contains_key(key) {
            out.insert(key.clone(), value.clone());
        }
    }
    Ok(Value::Object(out))
}

/// Serialize exactly like Python json.dumps(..., ensure_ascii=False): ", " between
/// entries, ": " after keys, unicode kept as-is. serde_json emits the compact form,
/// which tokenizes differently under Metaspace BPE.
fn py_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\x08' => out.push_str("\\b"),
                    '\x0c' => out.push_str("\\f"),
                    c if (c as u32) < 0x20 => {
                        out.push_str(&format!("\\u{:04x}", c as u32))
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push('"');
                out.push_str(key);
                out.push_str("\": ");
                py_json(item, out);
            }
            out.push('}');
        }
    }
}

impl VoyagPredictor {
    /// Load schema, tokenizer, and ONNX session from a release package directory.
    pub fn create(model_dir: &str) -> Result<Self, Box<dyn Error>> {
        let root = Path::new(model_dir);
        let schema: Value = serde_json::from_str(&std::fs::read_to_string(
            root.join("schema").join("voyag_schema.json"),
        )?)?;
        let special_map: Value = serde_json::from_str(&std::fs::read_to_string(
            root.join("tokenizer").join("special_tokens_map.json"),
        )?)?;

        let tokenizer =
            Tokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))
                .map_err(|e| format!("failed to load tokenizer: {e}"))?;

        let mask_token = special_map["mask_token"]["content"]
            .as_str()
            .ok_or("special_tokens_map.json has no mask_token")?
            .to_string();
        let special_id = |key: &str| -> Result<i64, Box<dyn Error>> {
            let content = special_map[key]["content"]
                .as_str()
                .ok_or(format!("special_tokens_map.json has no {key}"))?;
            tokenizer
                .token_to_id(content)
                .map(|id| id as i64)
                .ok_or_else(|| {
                    format!("tokenizer is missing required special token: {content}").into()
                })
        };
        let (pad_id, cls_id, sep_id, mask_id) = (
            special_id("pad_token")?,
            special_id("cls_token")?,
            special_id("sep_token")?,
            special_id("mask_token")?,
        );

        let slot_ids: Vec<String> = schema["slots"]
            .as_object()
            .ok_or("schema has no slots object")?
            .keys()
            .cloned()
            .collect();
        let relation_labels: Vec<String> = schema["relation_labels"]
            .as_array()
            .ok_or("schema has no relation_labels")?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        let max_len = schema["model_input"]["max_len"].as_u64().unwrap_or(512) as usize;
        let head_max_len =
            schema["model_input"]["head_max_len"].as_u64().unwrap_or(256) as usize;
        let truncation_max = tokenizer
            .get_truncation()
            .map(|t| t.max_length)
            .unwrap_or(usize::MAX);

        let mut pairs = Vec::new();
        for i in 0..slot_ids.len() {
            for j in (i + 1)..slot_ids.len() {
                pairs.push((i, j));
            }
        }

        let session = Session::builder()
            .map_err(|e| format!("failed to init ONNX Runtime: {e}"))?
            .commit_from_file(root.join("onnx").join("voyag-v8.onnx"))?;

        let mut predictor = VoyagPredictor {
            schema,
            tokenizer,
            session,
            slot_ids,
            relation_labels,
            max_len,
            head_max_len,
            truncation_max,
            pad_id,
            cls_id,
            sep_id,
            mask_id,
            mask_token,
            pairs,
            head_cache: Vec::new(),
        };
        // Slot question heads are identical for every request; encode once.
        predictor.head_cache = {
            let mut cache = Vec::new();
            let slot_ids = predictor.slot_ids.clone();
            for qid in &slot_ids {
                let question = predictor.schema["slots"][qid].clone();
                cache.push(predictor.build_head(&question));
            }
            cache
        };
        Ok(predictor)
    }

    fn encode(&self, text: &str, truncate: Option<usize>) -> Result<Vec<i64>, Box<dyn Error>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("tokenization failed: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        // The tokenizers crate applies tokenizer.json truncation automatically; this
        // explicit pass is a parity guard for the Python behavior.
        let ids: Vec<i64> = ids.into_iter().take(self.truncation_max).collect();
        Ok(match truncate {
            Some(n) => ids.into_iter().take(n).collect(),
            None => ids,
        })
    }

    fn render_options(&self, question: &Value) -> Vec<String> {
        let criteria = question["criteria"].as_object().expect("slot has no criteria");
        criteria
            .iter()
            .map(|(label, description)| {
                let description = description.as_str().unwrap_or("");
                if description.is_empty() {
                    label.clone()
                } else {
                    format!("{label}: {description}")
                }
            })
            .collect()
    }

    fn build_head(&self, question: &Value) -> (Vec<i64>, Vec<i64>) {
        let options = self.render_options(question);
        let instruction = question["instructions"]
            .as_str()
            .unwrap_or("")
            .replace(&self.mask_token, " ");
        let mut head_ids = self
            .encode(&format!("{} question: {instruction}", question["type"].as_str().unwrap_or("choice")), None)
            .expect("head encode failed");

        let mut option_ids: Vec<Vec<i64>> = options
            .iter()
            .map(|option| {
                let tokens = self
                    .encode(&format!(" {}", option.replace(&self.mask_token, " ")), Some(48))
                    .expect("option encode failed");
                std::iter::once(self.mask_id).chain(tokens).collect()
            })
            .collect();

        let mut option_budget =
            self.head_max_len as isize - option_ids.iter().map(|t| t.len() as isize).sum::<isize>();
        if option_budget < 16 {
            let per_option =
                ((self.head_max_len - 16) / option_ids.len().max(1)).max(4);
            for tokens in &mut option_ids {
                tokens.truncate(per_option);
            }
            option_budget = self.head_max_len as isize
                - option_ids.iter().map(|t| t.len() as isize).sum::<isize>();
        }

        let keep = (option_budget as usize).max(8);
        head_ids.truncate(keep.min(head_ids.len()));

        let mut ids = vec![self.cls_id];
        ids.extend_from_slice(&head_ids);
        ids.push(self.sep_id);
        let mut markers = Vec::new();
        for tokens in &option_ids {
            markers.push(ids.len() as i64);
            ids.extend_from_slice(tokens);
        }
        ids.push(self.sep_id);
        (ids, markers)
    }

    fn state_ids(&self, state: &Value) -> Result<Vec<i64>, Box<dyn Error>> {
        let canonical = canonicalize_state(state)?;
        let mut serialized = String::new();
        py_json(&canonical, &mut serialized);
        self.encode(&serialized.replace(&self.mask_token, " "), None)
    }

    /// Predict intents and relations for a full vehicle state (must contain utterance).
    pub fn predict_state(&self, state: &Value) -> Result<Value, Box<dyn Error>> {
        let state_ids = self.state_ids(state)?;
        let rows = self.head_cache.len();
        let mut row_ids = Vec::with_capacity(rows);
        let mut row_markers = Vec::with_capacity(rows);
        let mut max_markers = 0usize;
        for (head_ids, head_markers) in &self.head_cache {
            let room = self.max_len.saturating_sub(head_ids.len() + 1);
            let part_len = room.min(state_ids.len());
            let mut ids: Vec<i64> = Vec::with_capacity(head_ids.len() + part_len + 1);
            ids.extend_from_slice(head_ids);
            ids.extend_from_slice(&state_ids[..part_len]);
            ids.push(self.sep_id);
            ids.truncate(self.max_len);
            let markers: Vec<i64> = head_markers
                .iter()
                .copied()
                .filter(|&m| (m as usize) < self.max_len)
                .collect();
            max_markers = max_markers.max(markers.len());
            row_ids.push(ids);
            row_markers.push(markers);
        }

        let mut input_ids = vec![self.pad_id; rows * self.max_len];
        let mut attention_mask = vec![0i64; rows * self.max_len];
        let mut marker_pos = vec![0i64; rows * max_markers];
        let mut marker_mask = vec![false; rows * max_markers];
        let qtype = vec![0i64; rows];

        for (r, ids) in row_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                input_ids[r * self.max_len + i] = id;
                attention_mask[r * self.max_len + i] = 1;
            }
            for (m, &marker) in row_markers[r].iter().enumerate() {
                marker_pos[r * max_markers + m] = marker;
                marker_mask[r * max_markers + m] = true;
            }
        }

        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([rows, self.max_len], input_ids))?,
            "attention_mask" => Tensor::from_array(([rows, self.max_len], attention_mask))?,
            "marker_pos" => Tensor::from_array(([rows, max_markers], marker_pos))?,
            "marker_mask" => Tensor::from_array(([rows, max_markers], marker_mask))?,
            "qtype" => Tensor::from_array(([rows], qtype))?,
        ])?;

        let (_, intent_prob) = outputs["intent_probabilities"].try_extract_tensor::<f32>()?;
        let (_, relation_prob) = outputs["relation_probabilities"].try_extract_tensor::<f32>()?;

        let mut active: Map<String, Value> = Map::new();
        let mut intents: Vec<Value> = Vec::new();
        // intent_probabilities is [num_slots, max_options]: each row is padded to
        // the widest slot with zeros. argmax over the full row, exactly like Python.
        let num_cols = intent_prob.len() / self.slot_ids.len();
        for (slot_index, qid) in self.slot_ids.iter().enumerate() {
            let option_labels: Vec<String> = self.schema["slots"][qid]["criteria"]
                .as_object()
                .expect("slot has no criteria")
                .keys()
                .cloned()
                .collect();
            let start = slot_index * num_cols;
            let mut argmax = 0usize;
            for k in 1..num_cols {
                if intent_prob[start + k] > intent_prob[start + argmax] {
                    argmax = k;
                }
            }
            if argmax >= option_labels.len() {
                // Defensive fallback: padded columns are zero, so this never triggers.
                argmax = 0;
                for k in 1..option_labels.len() {
                    if intent_prob[start + k] > intent_prob[start + argmax] {
                        argmax = k;
                    }
                }
            }
            let label = &option_labels[argmax];
            if label == "none" {
                continue;
            }
            active.insert(qid.clone(), Value::String(label.clone()));
            intents.push(Value::Array(vec![
                Value::String(qid.clone()),
                Value::String(label.clone()),
            ]));
        }

        let mut relations: Vec<Value> = Vec::new();
        let mut mutex_violations: Vec<Value> = Vec::new();
        for (pair_index, &(left_index, right_index)) in self.pairs.iter().enumerate() {
            let left_qid = &self.slot_ids[left_index];
            let right_qid = &self.slot_ids[right_index];
            if !active.contains_key(left_qid) || !active.contains_key(right_qid) {
                continue;
            }
            let start = pair_index * 3;
            let mut argmax = 0usize;
            for k in 1..3 {
                if relation_prob[start + k] > relation_prob[start + argmax] {
                    argmax = k;
                }
            }
            let relation_type = &self.relation_labels[argmax];
            let confidence = (relation_prob[start + argmax] as f64 * 10000.0).round() / 10000.0;
            relations.push(serde_json::json!({
                "source_id": left_qid,
                "target_id": right_qid,
                "type": relation_type,
                "confidence": confidence,
            }));
            if relation_type == "mutually_exclusive" {
                mutex_violations.push(serde_json::json!({
                    "source_id": left_qid,
                    "target_id": right_qid,
                    "type": "mutually_exclusive",
                }));
            }
        }

        let execution_hint = if !mutex_violations.is_empty() {
            "clarify"
        } else if relations.iter().any(|r| r["type"] == "sequential") {
            "sequential"
        } else if relations.iter().any(|r| r["type"] == "parallel") {
            "parallel"
        } else if intents.len() > 1 {
            "unknown"
        } else {
            "single"
        };

        let mut out = Map::new();
        out.insert("utterance".into(), state["utterance"].clone());
        out.insert("model".into(), Value::String(MODEL_NAME.into()));
        out.insert("intents".into(), Value::Array(intents));
        out.insert("relations".into(), Value::Array(relations));
        out.insert("execution_hint".into(), Value::String(execution_hint.into()));
        out.insert("mutex_violations".into(), Value::Array(mutex_violations.clone()));
        out.insert(
            "clarification".into(),
            if mutex_violations.is_empty() {
                Value::Null
            } else {
                Value::String("检测到互斥意图，请确认先执行哪一个。".into())
            },
        );
        Ok(Value::Object(out))
    }

    /// Predict for an utterance, merged over the neutral vehicle state.
    pub fn predict(&self, utterance: &str, state: Option<&Value>) -> Result<Value, Box<dyn Error>> {
        let mut merged = neutral_state();
        if let Some(extra) = state {
            let obj = merged.as_object_mut().expect("neutral state is an object");
            if let Some(extra_obj) = extra.as_object() {
                for (key, value) in extra_obj {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }
        merged["utterance"] = Value::String(utterance.to_string());
        self.predict_state(&merged)
    }
}
