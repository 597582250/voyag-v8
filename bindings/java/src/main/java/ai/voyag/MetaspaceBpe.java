package ai.voyag;

import com.google.gson.stream.JsonReader;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal Metaspace BPE tokenizer that reproduces the Python `tokenizers`
 * library behavior for Voyag V8 (normalizer Replace " " -> "▁", Metaspace
 * pre-tokenizer with prepend_scheme="always", BPE merges, no added-token
 * matching -- Voyag inference texts never contain added tokens).
 *
 * <p>This is the same zero-native-dependency route as laya-java's hand-written
 * BPE: no JNI, loads {@code tokenizer.json} directly.
 */
public final class MetaspaceBpe {

    private final Map<String, Integer> vocab = new HashMap<>(256_000);
    private final Map<String, Integer> merges = new HashMap<>(256_000);
    private final String replacement;
    private final Map<String, Integer> specialIds = new HashMap<>();
    private int unkId = -1;
    /** tokenizer.json `truncation.max_length`; Python applies it on every encode. */
    private final int truncationMax;

    private MetaspaceBpe(String replacement, int truncationMax) {
        this.replacement = replacement;
        this.truncationMax = truncationMax;
    }

    public static MetaspaceBpe fromFile(Path tokenizerJsonPath) throws IOException {
        try (Reader reader = Files.newBufferedReader(tokenizerJsonPath, StandardCharsets.UTF_8);
             JsonReader json = new JsonReader(reader)) {
            // Collect everything first; the JSON member order is not fixed.
            String replacement = "▁";
            String unkToken = "<unk>";
            Map<String, Integer> vocab = null;
            List<String> mergeList = null;
            Map<String, Integer> added = null;
            int truncationMax = Integer.MAX_VALUE;

            json.beginObject();
            while (json.hasNext()) {
                switch (json.nextName()) {
                    case "truncation" -> truncationMax = readTruncationMax(json);
                    case "pre_tokenizer" -> replacement = readMetaspaceReplacement(json);
                    case "model" -> {
                        json.beginObject();
                        while (json.hasNext()) {
                            switch (json.nextName()) {
                                case "unk_token" -> unkToken = json.nextString();
                                case "vocab" -> {
                                    vocab = new HashMap<>(256_000);
                                    json.beginObject();
                                    while (json.hasNext()) {
                                        vocab.put(json.nextName(), json.nextInt());
                                    }
                                    json.endObject();
                                }
                                case "merges" -> {
                                    mergeList = new ArrayList<>(256_000);
                                    json.beginArray();
                                    while (json.hasNext()) {
                                        mergeList.add(json.nextString());
                                    }
                                    json.endArray();
                                }
                                default -> json.skipValue();
                            }
                        }
                        json.endObject();
                    }
                    case "added_tokens" -> {
                        added = new HashMap<>();
                        json.beginArray();
                        while (json.hasNext()) {
                            String content = null;
                            int id = -1;
                            json.beginObject();
                            while (json.hasNext()) {
                                switch (json.nextName()) {
                                    case "content" -> content = json.nextString();
                                    case "id" -> id = json.nextInt();
                                    default -> json.skipValue();
                                }
                            }
                            json.endObject();
                            if (content != null && id >= 0) {
                                added.put(content, id);
                            }
                        }
                        json.endArray();
                    }
                    default -> json.skipValue();
                }
            }
            json.endObject();

            if (vocab == null || mergeList == null) {
                throw new IOException("tokenizer.json has no BPE model");
            }
            MetaspaceBpe tokenizer = new MetaspaceBpe(replacement, truncationMax);
            tokenizer.vocab.putAll(vocab);
            for (int rank = 0; rank < mergeList.size(); rank++) {
                tokenizer.merges.put(mergeList.get(rank), rank);
            }
            if (added != null) {
                tokenizer.specialIds.putAll(added);
                Integer unk = added.get(unkToken);
                if (unk != null) {
                    tokenizer.unkId = unk;
                }
            }
            return tokenizer;
        }
    }

    private static int readTruncationMax(JsonReader json) throws IOException {
        int maxLength = Integer.MAX_VALUE;
        json.beginObject();
        while (json.hasNext()) {
            if (json.nextName().equals("max_length")) {
                maxLength = json.nextInt();
            } else {
                json.skipValue();
            }
        }
        json.endObject();
        return maxLength;
    }

    private static String readMetaspaceReplacement(JsonReader json) throws IOException {
        String replacement = "▁";
        json.beginObject();
        while (json.hasNext()) {
            if (json.nextName().equals("replacement")) {
                replacement = json.nextString();
            } else {
                json.skipValue();
            }
        }
        json.endObject();
        return replacement;
    }

    public int idOfSpecial(String content) {
        Integer id = specialIds.get(content);
        if (id == null) {
            throw new IllegalArgumentException("tokenizer is missing required special token: " + content);
        }
        return id;
    }

    public int unkId() {
        return unkId;
    }

    /** Encode text without special tokens, truncating to {@code truncate} ids when non-null.
     * The tokenizer-level truncation max_length is always applied first. */
    public int[] encode(String text, Integer truncate) {
        String normalized = text.replace(" ", replacement);
        String prepared = replacement + normalized;
        List<Integer> ids = new ArrayList<>();
        for (String word : prepared.split(java.util.regex.Pattern.quote(replacement), -1)) {
            if (word.isEmpty()) {
                continue;
            }
            encodeWord(replacement + word, ids);
        }
        int effectiveLimit = ids.size();
        if (effectiveLimit > truncationMax) {
            effectiveLimit = truncationMax;
        }
        if (truncate != null && effectiveLimit > truncate) {
            effectiveLimit = truncate;
        }
        int[] result = new int[effectiveLimit];
        for (int i = 0; i < result.length; i++) {
            result[i] = ids.get(i);
        }
        return result;
    }

    private void encodeWord(String word, List<Integer> ids) {
        // Split into Unicode code points, then greedily merge by merge rank.
        List<String> symbols = new ArrayList<>();
        word.codePoints().forEach(cp -> symbols.add(new String(Character.toChars(cp))));

        while (true) {
            int bestRank = Integer.MAX_VALUE;
            int bestIndex = -1;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = merges.get(symbols.get(i) + " " + symbols.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) {
                break;
            }
            symbols.set(bestIndex, symbols.get(bestIndex) + symbols.get(bestIndex + 1));
            symbols.remove(bestIndex + 1);
        }

        for (String symbol : symbols) {
            Integer id = vocab.get(symbol);
            ids.add(id == null ? unkId : id);
        }
    }
}
