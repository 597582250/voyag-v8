package ai.voyag;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.FloatBuffer;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Voyag V8 local inference for Java.
 *
 * <p>Reproduces the exact pipeline of the Python runner {@code inference.py}:
 * schema + tokenizer (bundled {@link MetaspaceBpe}) + fixed [10, 512] ONNX
 * input + decode. Depends only on ai.onnxruntime and Gson; no PyTorch.
 */
public final class VoyagPredictor implements AutoCloseable {

    private static final String MODEL_NAME = "voyag-v8-onnx";

    private static final List<String> CANONICAL_STATE_FIELDS = List.of(
            "utterance", "speed_kmh", "acc", "window_state", "hvac_state",
            "hvac_temperature_c", "media_state", "seat_heat", "light_state",
            "phone_call", "nav_active", "battery_soc", "gps_valid", "dtc", "warning");

    private static LinkedHashMap<String, Object> neutralState() {
        LinkedHashMap<String, Object> state = new LinkedHashMap<>();
        state.put("utterance", "");
        state.put("speed_kmh", 0L);
        state.put("acc", "off");
        state.put("window_state", "closed");
        state.put("hvac_state", "off");
        state.put("hvac_temperature_c", 22L);
        state.put("media_state", "idle");
        state.put("seat_heat", "off");
        state.put("light_state", "off");
        state.put("phone_call", "idle");
        state.put("nav_active", false);
        state.put("battery_soc", 0.7d);
        state.put("gps_valid", true);
        state.put("dtc", new ArrayList<Object>());
        state.put("warning", new ArrayList<Object>());
        return state;
    }

    private record Head(int[] ids, int[] markers) {}

    private record Pair(int left, int right) {}

    private final JsonObject schema;
    private final MetaspaceBpe tokenizer;
    private final OrtSession session;
    private final OrtEnvironment env;
    private final long padId, clsId, sepId, maskId;
    private final String maskToken;
    private final int maxLen, headMaxLen;
    private final List<String> slotIds;
    private final List<String> relationLabels;
    private final List<Pair> pairs = new ArrayList<>();
    /** Pre-encoded slot-question heads: identical for every request. */
    private final List<Head> headCache = new ArrayList<>();

    private VoyagPredictor(JsonObject schema, MetaspaceBpe tokenizer, OrtSession session,
                           OrtEnvironment env, long padId, long clsId, long sepId,
                           long maskId, String maskToken) {
        this.schema = schema;
        this.tokenizer = tokenizer;
        this.session = session;
        this.env = env;
        this.padId = padId;
        this.clsId = clsId;
        this.sepId = sepId;
        this.maskId = maskId;
        this.maskToken = maskToken;

        JsonObject slots = schema.getAsJsonObject("slots");
        this.slotIds = new ArrayList<>(slots.keySet());
        this.relationLabels = new ArrayList<>();
        for (JsonElement label : schema.getAsJsonArray("relation_labels")) {
            this.relationLabels.add(label.getAsString());
        }
        JsonObject modelInput = schema.getAsJsonObject("model_input");
        this.maxLen = modelInput.get("max_len").getAsInt();
        this.headMaxLen = modelInput.get("head_max_len").getAsInt();

        for (int i = 0; i < slotIds.size(); i++) {
            for (int j = i + 1; j < slotIds.size(); j++) {
                pairs.add(new Pair(i, j));
            }
        }
        // Slot question heads are identical for every request; encode once.
        for (String qid : slotIds) {
            headCache.add(buildHead(slots.getAsJsonObject(qid)));
        }
    }

    /** Load schema, tokenizer, and ONNX session from a release package directory. */
    public static VoyagPredictor create(String modelDir) throws IOException, OrtException {
        Path root = Paths.get(modelDir);
        JsonObject schema = JsonParser.parseString(Files.readString(
                root.resolve("schema").resolve("voyag_schema.json"),
                StandardCharsets.UTF_8)).getAsJsonObject();
        MetaspaceBpe tokenizer = MetaspaceBpe.fromFile(
                root.resolve("tokenizer").resolve("tokenizer.json"));
        JsonObject specialMap = JsonParser.parseString(Files.readString(
                root.resolve("tokenizer").resolve("special_tokens_map.json"),
                StandardCharsets.UTF_8)).getAsJsonObject();

        String maskToken = specialMap.getAsJsonObject("mask_token").get("content").getAsString();
        long padId = specialId(tokenizer, specialMap, "pad_token");
        long clsId = specialId(tokenizer, specialMap, "cls_token");
        long sepId = specialId(tokenizer, specialMap, "sep_token");
        long maskId = specialId(tokenizer, specialMap, "mask_token");

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        String threads = System.getenv("VOYAG_THREADS");
        if (threads != null && !threads.isEmpty()) {
            options.setIntraOpNumThreads(Integer.parseInt(threads));
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(
                root.resolve("onnx").resolve("voyag-v8.onnx").toString(), options);
        return new VoyagPredictor(schema, tokenizer, session, env,
                padId, clsId, sepId, maskId, maskToken);
    }

    private static long specialId(MetaspaceBpe tokenizer, JsonObject specialMap, String key) {
        String content = specialMap.getAsJsonObject(key).get("content").getAsString();
        return tokenizer.idOfSpecial(content);
    }

    private int[] encode(String text, Integer truncate) {
        return tokenizer.encode(text, truncate);
    }

    private static List<String> renderOptions(JsonObject question) {
        List<String> options = new ArrayList<>();
        for (Map.Entry<String, JsonElement> entry
                : question.getAsJsonObject("criteria").entrySet()) {
            String label = entry.getKey();
            String description = entry.getValue().isJsonNull()
                    ? "" : entry.getValue().getAsString();
            options.add(description.isEmpty() ? label : label + ": " + description);
        }
        return options;
    }

    private Head buildHead(JsonObject question) {
        List<String> options = renderOptions(question);
        String instruction = question.get("instructions").getAsString()
                .replace(maskToken, " ");
        int[] headIds = encode(
                question.get("type").getAsString() + " question: " + instruction, null);

        List<int[]> optionIds = new ArrayList<>();
        for (String option : options) {
            int[] tokens = encode(" " + option.replace(maskToken, " "), 48);
            int[] withMask = new int[tokens.length + 1];
            withMask[0] = (int) maskId;
            System.arraycopy(tokens, 0, withMask, 1, tokens.length);
            optionIds.add(withMask);
        }

        int optionBudget = headMaxLen - optionIds.stream().mapToInt(t -> t.length).sum();
        if (optionBudget < 16) {
            int perOption = Math.max(4, (headMaxLen - 16) / Math.max(1, optionIds.size()));
            for (int i = 0; i < optionIds.size(); i++) {
                optionIds.set(i, Arrays.copyOf(optionIds.get(i),
                        Math.min(perOption, optionIds.get(i).length)));
            }
            optionBudget = headMaxLen - optionIds.stream().mapToInt(t -> t.length).sum();
        }

        headIds = Arrays.copyOf(headIds, Math.max(8, Math.min(optionBudget, headIds.length)));
        List<Integer> ids = new ArrayList<>();
        ids.add((int) clsId);
        for (int id : headIds) ids.add(id);
        ids.add((int) sepId);
        List<Integer> markers = new ArrayList<>();
        for (int[] tokens : optionIds) {
            markers.add(ids.size());
            for (int token : tokens) ids.add(token);
        }
        ids.add((int) sepId);
        return new Head(toIntArray(ids), toIntArray(markers));
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] result = new int[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = values.get(i);
        return result;
    }

    /** Serialize exactly like Python json.dumps(..., ensure_ascii=False). */
    private static String serializeState(LinkedHashMap<String, Object> state) {
        StringBuilder sb = new StringBuilder("{");
        int index = 0;
        for (Map.Entry<String, Object> entry : state.entrySet()) {
            if (index++ > 0) sb.append(", ");
            sb.append('"').append(entry.getKey()).append("\": ");
            appendJsonValue(sb, entry.getValue());
        }
        sb.append('}');
        return sb.toString();
    }

    private static void appendJsonValue(StringBuilder sb, Object value) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String s) {
            sb.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    case '\b' -> sb.append("\\b");
                    case '\f' -> sb.append("\\f");
                    default -> {
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                    }
                }
            }
            sb.append('"');
        } else if (value instanceof Boolean b) {
            sb.append(b ? "true" : "false");
        } else if (value instanceof Byte || value instanceof Short
                || value instanceof Integer || value instanceof Long) {
            sb.append(((Number) value).longValue());
        } else if (value instanceof Number number) {
            // Python repr parity holds for the magnitudes used in vehicle state.
            sb.append(number.doubleValue());
        } else if (value instanceof List<?> list) {
            sb.append('[');
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(", ");
                appendJsonValue(sb, list.get(i));
            }
            sb.append(']');
        } else {
            throw new IllegalArgumentException(
                    "unsupported state value type: " + value.getClass());
        }
    }

    private static LinkedHashMap<String, Object> canonicalizeState(Map<String, Object> state) {
        if (!state.containsKey("utterance")) {
            throw new IllegalArgumentException("state must contain 'utterance'");
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        for (String key : CANONICAL_STATE_FIELDS) {
            if (state.containsKey(key)) out.put(key, state.get(key));
        }
        for (Map.Entry<String, Object> entry : state.entrySet()) {
            if (!out.containsKey(entry.getKey())) out.put(entry.getKey(), entry.getValue());
        }
        return out;
    }

    private int[] stateIds(LinkedHashMap<String, Object> state) {
        String serialized = serializeState(state);
        return encode(serialized.replace(maskToken, " "), null);
    }

    private Map<String, OnnxTensor> buildInputs(LinkedHashMap<String, Object> state)
            throws OrtException {
        int[] stateIds = stateIds(state);
        int rows = slotIds.size();
        List<int[]> rowIds = new ArrayList<>();
        List<int[]> rowMarkers = new ArrayList<>();
        int maxMarkers = 0;
        for (Head head : headCache) {
            int room = Math.max(0, maxLen - head.ids().length - 1);
            int partLen = Math.min(room, stateIds.length);
            int[] ids = new int[head.ids().length + partLen + 1];
            System.arraycopy(head.ids(), 0, ids, 0, head.ids().length);
            System.arraycopy(stateIds, 0, ids, head.ids().length, partLen);
            ids[ids.length - 1] = (int) sepId;
            if (ids.length > maxLen) ids = Arrays.copyOf(ids, maxLen);
            int[] markers = Arrays.stream(head.markers())
                    .filter(m -> m < maxLen).toArray();
            rowIds.add(ids);
            rowMarkers.add(markers);
            maxMarkers = Math.max(maxMarkers, markers.length);
        }

        long[] inputIds = new long[rows * maxLen];
        Arrays.fill(inputIds, padId);
        long[] attentionMask = new long[rows * maxLen];
        long[] markerPos = new long[rows * maxMarkers];
        boolean[] markerMask = new boolean[rows * maxMarkers];
        long[] qtype = new long[rows];

        for (int r = 0; r < rows; r++) {
            int[] ids = rowIds.get(r);
            for (int i = 0; i < ids.length; i++) {
                inputIds[r * maxLen + i] = ids[i];
                attentionMask[r * maxLen + i] = 1;
            }
            int[] markers = rowMarkers.get(r);
            for (int m = 0; m < markers.length; m++) {
                markerPos[r * maxMarkers + m] = markers[m];
                markerMask[r * maxMarkers + m] = true;
            }
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        byte[] markerMaskBytes = new byte[markerMask.length];
        for (int i = 0; i < markerMask.length; i++) {
            markerMaskBytes[i] = markerMask[i] ? (byte) 1 : (byte) 0;
        }
        inputs.put("input_ids",
                OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), new long[]{rows, maxLen}));
        inputs.put("attention_mask",
                OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), new long[]{rows, maxLen}));
        inputs.put("marker_pos",
                OnnxTensor.createTensor(env, LongBuffer.wrap(markerPos), new long[]{rows, maxMarkers}));
        inputs.put("marker_mask",
                OnnxTensor.createTensor(env, ByteBuffer.wrap(markerMaskBytes),
                        new long[]{rows, maxMarkers}, OnnxJavaType.BOOL));
        inputs.put("qtype",
                OnnxTensor.createTensor(env, LongBuffer.wrap(qtype), new long[]{rows}));
        return inputs;
    }

    /** Predict intents and relations for a full vehicle state (must contain utterance). */
    public Map<String, Object> predictState(Map<String, Object> state) throws OrtException {
        LinkedHashMap<String, Object> canonical = canonicalizeState(state);
        Map<String, OnnxTensor> inputs = buildInputs(canonical);
        try (OrtSession.Result result = session.run(inputs)) {
            OnnxTensor intentTensor =
                    (OnnxTensor) result.get("intent_probabilities").get();
            OnnxTensor relationTensor =
                    (OnnxTensor) result.get("relation_probabilities").get();
            FloatBuffer intentProb = intentTensor.getFloatBuffer();
            FloatBuffer relationProb = relationTensor.getFloatBuffer();

            Map<String, String> active = new LinkedHashMap<>();
            List<List<String>> intents = new ArrayList<>();
            // intent_probabilities is [num_slots, max_options]: each row is padded
            // to the widest slot with zeros. argmax over the full row, like Python.
            int numCols = (int) intentTensor.getInfo().getShape()[1];
            for (int slotIndex = 0; slotIndex < slotIds.size(); slotIndex++) {
                JsonObject criteria = schema.getAsJsonObject("slots")
                        .getAsJsonObject(slotIds.get(slotIndex)).getAsJsonObject("criteria");
                List<String> optionLabels = new ArrayList<>(criteria.keySet());
                int start = slotIndex * numCols;
                int argmax = 0;
                for (int k = 1; k < numCols; k++) {
                    if (intentProb.get(start + k) > intentProb.get(start + argmax)) argmax = k;
                }
                if (argmax >= optionLabels.size()) {
                    // Defensive fallback: padded columns are zero, so this never triggers.
                    argmax = 0;
                    for (int k = 1; k < optionLabels.size(); k++) {
                        if (intentProb.get(start + k) > intentProb.get(start + argmax)) argmax = k;
                    }
                }
                String label = optionLabels.get(argmax);
                if ("none".equals(label)) continue;
                active.put(slotIds.get(slotIndex), label);
                intents.add(List.of(slotIds.get(slotIndex), label));
            }

            List<Map<String, Object>> relations = new ArrayList<>();
            List<Map<String, Object>> mutexViolations = new ArrayList<>();
            for (int pairIndex = 0; pairIndex < pairs.size(); pairIndex++) {
                Pair pair = pairs.get(pairIndex);
                String leftQid = slotIds.get(pair.left());
                String rightQid = slotIds.get(pair.right());
                if (!active.containsKey(leftQid) || !active.containsKey(rightQid)) continue;
                int start = pairIndex * 3;
                int argmax = 0;
                for (int k = 1; k < 3; k++) {
                    if (relationProb.get(start + k) > relationProb.get(start + argmax)) argmax = k;
                }
                String relationType = relationLabels.get(argmax);
                Map<String, Object> relation = new LinkedHashMap<>();
                relation.put("source_id", leftQid);
                relation.put("target_id", rightQid);
                relation.put("type", relationType);
                relation.put("confidence",
                        Math.round(relationProb.get(start + argmax) * 10000) / 10000.0);
                relations.add(relation);
                if ("mutually_exclusive".equals(relationType)) {
                    Map<String, Object> violation = new LinkedHashMap<>();
                    violation.put("source_id", leftQid);
                    violation.put("target_id", rightQid);
                    violation.put("type", "mutually_exclusive");
                    mutexViolations.add(violation);
                }
            }

            String executionHint;
            if (!mutexViolations.isEmpty()) executionHint = "clarify";
            else if (relations.stream().anyMatch(r -> "sequential".equals(r.get("type"))))
                executionHint = "sequential";
            else if (relations.stream().anyMatch(r -> "parallel".equals(r.get("type"))))
                executionHint = "parallel";
            else if (intents.size() > 1) executionHint = "unknown";
            else executionHint = "single";

            Map<String, Object> output = new LinkedHashMap<>();
            output.put("utterance", canonical.getOrDefault("utterance", ""));
            output.put("model", MODEL_NAME);
            output.put("intents", intents);
            output.put("relations", relations);
            output.put("execution_hint", executionHint);
            output.put("mutex_violations", mutexViolations);
            output.put("clarification", mutexViolations.isEmpty()
                    ? null : "检测到互斥意图，请确认先执行哪一个。");
            return output;
        }
    }

    /** Predict for an utterance, merged over the neutral vehicle state. */
    public Map<String, Object> predict(String utterance, Map<String, Object> state)
            throws OrtException {
        LinkedHashMap<String, Object> merged = neutralState();
        if (state != null) merged.putAll(state);
        merged.put("utterance", utterance);
        return predictState(merged);
    }

    @Override
    public void close() {
        session.close();
        // OrtEnvironment.getEnvironment() is a shared singleton; never close it.
    }
}
