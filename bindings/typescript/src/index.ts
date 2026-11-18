/**
 * Voyag V8 local inference for TypeScript / Node.js.
 *
 * Reproduces the exact pipeline of the Python runner `inference.py`:
 * schema + tokenizer (Metaspace BPE) + fixed [10, 512] ONNX input + decode.
 */
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { AutoTokenizer, env } from "@huggingface/transformers";
import * as ort from "onnxruntime-node";

env.allowLocalModels = true;

const MODEL_NAME = "voyag-v8-onnx";

/**
 * Serialize a JSON value exactly like Python json.dumps(..., ensure_ascii=False):
 * ", " between entries, ": " after keys, unicode kept as-is. JSON.stringify emits
 * the compact form ("," / ":"), which tokenizes differently under Metaspace BPE.
 */
function pythonJson(value: unknown): string {
  if (value === null || typeof value === "number" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return "[" + value.map(pythonJson).join(", ") + "]";
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${JSON.stringify(key)}: ${pythonJson(item)}`);
    return "{" + entries.join(", ") + "}";
  }
  throw new Error(`unsupported state value type: ${typeof value}`);
}

const CANONICAL_STATE_FIELDS = [
  "utterance", "speed_kmh", "acc", "window_state", "hvac_state",
  "hvac_temperature_c", "media_state", "seat_heat", "light_state",
  "phone_call", "nav_active", "battery_soc", "gps_valid", "dtc", "warning",
] as const;

const NEUTRAL_STATE: Record<string, unknown> = {
  utterance: "", speed_kmh: 0, acc: "off", window_state: "closed",
  hvac_state: "off", hvac_temperature_c: 22, media_state: "idle",
  seat_heat: "off", light_state: "off", phone_call: "idle",
  nav_active: false, battery_soc: 0.7, gps_valid: true, dtc: [], warning: [],
};

interface SlotSchema {
  type: string;
  instructions: string;
  criteria: Record<string, string>;
}

interface VoyagSchema {
  slots: Record<string, SlotSchema>;
  relation_labels: string[];
  model_input: { max_len: number; head_max_len: number; num_slots: number };
}

/** Resolve a special token id from the tokenizer.json added_tokens table. */
function addedTokenIds(tokenizerJson: any): Map<string, number> {
  const ids = new Map<string, number>();
  for (const t of tokenizerJson.added_tokens ?? []) {
    ids.set(t.content, t.id);
  }
  return ids;
}

/** Metaspace BPE fallback encode via tokenizers.js (identical to Python). */
export class VoyagPredictor {
  private schema: VoyagSchema;
  private tokenizer: any;
  private session: ort.InferenceSession;
  /** special_tokens_map.json content, e.g. {"pad_token": {"content": "<pad>"}} */
  private specialMap: Record<string, { content: string }>;
  private special: Map<string, number>;
  private padId: number;
  private clsId: number;
  private sepId: number;
  private maskId: number;
  private maskToken: string;
  private maxLen: number;
  private headMaxLen: number;
  /** tokenizer.json `truncation.max_length`; Python applies it on every encode. */
  private truncationMax: number;
  private slotIds: string[];
  private pairs: Array<[number, number]>;
  /** Pre-encoded slot-question heads: [ids, markers][] */
  private headCache: Array<{ ids: number[]; markers: number[] }>;

  private constructor(
    schema: VoyagSchema,
    tokenizer: any,
    session: ort.InferenceSession,
    specialMap: Record<string, { content: string }>,
    special: Map<string, number>,
    truncationMax: number,
  ) {
    this.schema = schema;
    this.tokenizer = tokenizer;
    this.session = session;
    this.specialMap = specialMap;
    this.special = special;
    this.maskToken = specialMap.mask_token.content;
    this.padId = this.specialId("pad_token");
    this.clsId = this.specialId("cls_token");
    this.sepId = this.specialId("sep_token");
    this.maskId = this.specialId("mask_token");
    this.maxLen = schema.model_input.max_len;
    this.headMaxLen = schema.model_input.head_max_len;
    this.truncationMax = truncationMax;
    this.slotIds = Object.keys(schema.slots);
    this.pairs = [];
    for (let i = 0; i < this.slotIds.length; i++) {
      for (let j = i + 1; j < this.slotIds.length; j++) this.pairs.push([i, j]);
    }
    // Slot question heads are identical for every request; encode once.
    this.headCache = this.slotIds.map((qid) => this.buildHead(schema.slots[qid]));
  }

  static async create(modelDir: string): Promise<VoyagPredictor> {
    const dir = resolve(modelDir);
    const schema = JSON.parse(
      readFileSync(join(dir, "schema", "voyag_schema.json"), "utf-8"),
    ) as VoyagSchema;
    const tokenizerJson = JSON.parse(
      readFileSync(join(dir, "tokenizer", "tokenizer.json"), "utf-8"),
    );
    const specialMap = JSON.parse(
      readFileSync(join(dir, "tokenizer", "special_tokens_map.json"), "utf-8"),
    );
    const tokenizer = await AutoTokenizer.from_pretrained(join(dir, "tokenizer"));
    const session = await ort.InferenceSession.create(
      join(dir, "onnx", "voyag-v8.onnx"),
      { executionProviders: ["cpu"] },
    );
    const truncationMax = tokenizerJson.truncation?.max_length ?? Number.MAX_SAFE_INTEGER;
    return new VoyagPredictor(
      schema, tokenizer, session, specialMap, addedTokenIds(tokenizerJson), truncationMax,
    );
  }

  get device(): string {
    // The session is always created with executionProviders: ["cpu"].
    return "cpu";
  }

  private specialId(key: string): number {
    const content = this.specialMap[key].content;
    const id = this.special.get(content);
    if (id === undefined) throw new Error(`tokenizer is missing required special token: ${content}`);
    return id;
  }

  private encode(text: string, truncate?: number): number[] {
    let ids = this.tokenizer.encode(text, { add_special_tokens: false }) as number[];
    // tokenizer.json sets truncation max_length; Python applies it on every encode.
    if (ids.length > this.truncationMax) ids = ids.slice(0, this.truncationMax);
    return truncate === undefined ? ids : ids.slice(0, truncate);
  }

  private renderOptions(question: SlotSchema): string[] {
    return Object.entries(question.criteria).map(([label, description]) =>
      description === "" || description === null ? label : `${label}: ${description}`,
    );
  }

  private buildHead(question: SlotSchema): { ids: number[]; markers: number[] } {
    const options = this.renderOptions(question);
    const instruction = String(question.instructions).split(this.maskToken).join(" ");
    let headIds = this.encode(`${question.type} question: ${instruction}`);

    let optionIds = options.map((option) => {
      const tokens = this.encode(
        " " + option.split(this.maskToken).join(" "),
        48,
      );
      return [this.maskId, ...tokens];
    });

    let optionBudget = this.headMaxLen - optionIds.reduce((sum, t) => sum + t.length, 0);
    if (optionBudget < 16) {
      const perOption = Math.max(4, Math.floor((this.headMaxLen - 16) / Math.max(1, optionIds.length)));
      optionIds = optionIds.map((tokens) => tokens.slice(0, perOption));
      optionBudget = this.headMaxLen - optionIds.reduce((sum, t) => sum + t.length, 0);
    }

    headIds = headIds.slice(0, Math.max(8, optionBudget));
    const ids = [this.clsId, ...headIds, this.sepId];
    const markers: number[] = [];
    for (const tokens of optionIds) {
      markers.push(ids.length);
      ids.push(...tokens);
    }
    ids.push(this.sepId);
    return { ids, markers };
  }

  private canonicalizeState(state: Record<string, unknown>): Record<string, unknown> {
    if (!("utterance" in state)) throw new Error("state must contain 'utterance'");
    const out: Record<string, unknown> = {};
    for (const key of CANONICAL_STATE_FIELDS) {
      if (key in state) out[key] = state[key];
    }
    for (const [key, value] of Object.entries(state)) {
      if (!(key in out)) out[key] = value;
    }
    return out;
  }

  private stateIds(state: Record<string, unknown>): number[] {
    const serialized = pythonJson(this.canonicalizeState(state));
    return this.encode(serialized.split(this.maskToken).join(" "));
  }

  private buildInputs(state: Record<string, unknown>): Record<string, ort.Tensor> {
    const statePart = this.stateIds(state);
    const rows = this.headCache.map(({ ids: headIds, markers: headMarkers }) => {
      const room = Math.max(0, this.maxLen - headIds.length - 1);
      const part = statePart.slice(0, room);
      const ids = [...headIds, ...part, this.sepId].slice(0, this.maxLen);
      const markers = headMarkers.filter((m) => m < this.maxLen);
      return { ids, markers };
    });

    const maxMarkers = Math.max(...rows.map((r) => r.markers.length));
    const inputIds = new BigInt64Array(rows.length * this.maxLen).fill(BigInt(this.padId));
    const attentionMask = new BigInt64Array(rows.length * this.maxLen);
    const markerPos = new BigInt64Array(rows.length * maxMarkers);
    const markerMask = new Uint8Array(rows.length * maxMarkers);
    const qtype = new BigInt64Array(rows.length);

    rows.forEach((row, rowIdx) => {
      row.ids.forEach((id, i) => {
        inputIds[rowIdx * this.maxLen + i] = BigInt(id);
        attentionMask[rowIdx * this.maxLen + i] = 1n;
      });
      row.markers.forEach((m, i) => {
        markerPos[rowIdx * maxMarkers + i] = BigInt(m);
        markerMask[rowIdx * maxMarkers + i] = 1;
      });
    });

    return {
      input_ids: new ort.Tensor("int64", inputIds, [rows.length, this.maxLen]),
      attention_mask: new ort.Tensor("int64", attentionMask, [rows.length, this.maxLen]),
      marker_pos: new ort.Tensor("int64", markerPos, [rows.length, maxMarkers]),
      marker_mask: new ort.Tensor("bool", markerMask, [rows.length, maxMarkers]),
      qtype: new ort.Tensor("int64", qtype, [rows.length]),
    };
  }

  async predictState(state: Record<string, unknown>): Promise<Record<string, unknown>> {
    state = this.canonicalizeState({ ...state });
    const feeds = this.buildInputs(state);
    const outputs = await this.session.run(feeds as Record<string, ort.Tensor>);
    const intentProb = outputs.intent_probabilities.data as Float32Array;
    const relationProb = outputs.relation_probabilities.data as Float32Array;

    const active = new Map<string, string>();
    const intents: Array<[string, string]> = [];
    // intent_probabilities is [num_slots, max_options]: each row is padded to
    // the widest slot with zeros. argmax over the full row, exactly like Python.
    const numCols = intentProb.length / this.slotIds.length;
    this.slotIds.forEach((qid, slotIndex) => {
      const optionLabels = Object.keys(this.schema.slots[qid].criteria);
      const start = slotIndex * numCols;
      let argmax = 0;
      for (let k = 1; k < numCols; k++) {
        if (intentProb[start + k] > intentProb[start + argmax]) argmax = k;
      }
      if (argmax >= optionLabels.length) {
        // Defensive fallback: padded columns are zero, so this never triggers.
        argmax = 0;
        for (let k = 1; k < optionLabels.length; k++) {
          if (intentProb[start + k] > intentProb[start + argmax]) argmax = k;
        }
      }
      const label = optionLabels[argmax];
      if (label === "none") return;
      active.set(qid, label);
      intents.push([qid, label]);
    });

    const relations: Array<Record<string, unknown>> = [];
    const mutexViolations: Array<Record<string, unknown>> = [];
    this.pairs.forEach(([leftIndex, rightIndex], pairIndex) => {
      const leftQid = this.slotIds[leftIndex];
      const rightQid = this.slotIds[rightIndex];
      if (!active.has(leftQid) || !active.has(rightQid)) return;
      const start = pairIndex * 3;
      let argmax = 0;
      for (let k = 1; k < 3; k++) {
        if (relationProb[start + k] > relationProb[start + argmax]) argmax = k;
      }
      const relationType = this.schema.relation_labels[argmax];
      const relation = {
        source_id: leftQid, target_id: rightQid, type: relationType,
        confidence: Math.round(relationProb[start + argmax] * 10000) / 10000,
      };
      relations.push(relation);
      if (relationType === "mutually_exclusive") {
        mutexViolations.push({
          source_id: leftQid, target_id: rightQid, type: "mutually_exclusive",
        });
      }
    });

    let executionHint: string;
    if (mutexViolations.length > 0) executionHint = "clarify";
    else if (relations.some((r) => r.type === "sequential")) executionHint = "sequential";
    else if (relations.some((r) => r.type === "parallel")) executionHint = "parallel";
    else if (intents.length > 1) executionHint = "unknown";
    else executionHint = "single";

    return {
      utterance: (state as any).utterance ?? "",
      model: MODEL_NAME,
      intents,
      relations,
      execution_hint: executionHint,
      mutex_violations: mutexViolations,
      clarification: mutexViolations.length > 0
        ? "检测到互斥意图，请确认先执行哪一个。" : null,
    };
  }

  async predict(utterance: string, state?: Record<string, unknown>): Promise<Record<string, unknown>> {
    const merged: Record<string, unknown> = { ...NEUTRAL_STATE, ...(state ?? {}) };
    merged.utterance = utterance;
    return this.predictState(merged);
  }
}
