// Voyag V8 local inference for C# / .NET.
//
// Reproduces the exact pipeline of the Python runner `inference.py`:
// schema + tokenizer (Metaspace BPE via Tokenizers.DotNet) + fixed
// [10, 512] ONNX input + decode.
using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.DotNet;

namespace Voyag;

public sealed class VoyagPredictor : IDisposable
{
    private const string ModelName = "voyag-v8-onnx";

    private static readonly string[] CanonicalStateFields =
    {
        "utterance", "speed_kmh", "acc", "window_state", "hvac_state",
        "hvac_temperature_c", "media_state", "seat_heat", "light_state",
        "phone_call", "nav_active", "battery_soc", "gps_valid", "dtc", "warning",
    };

    private static readonly List<KeyValuePair<string, object?>> NeutralState = new()
    {
        new("utterance", ""), new("speed_kmh", 0L), new("acc", "off"),
        new("window_state", "closed"), new("hvac_state", "off"),
        new("hvac_temperature_c", 22L), new("media_state", "idle"),
        new("seat_heat", "off"), new("light_state", "off"),
        new("phone_call", "idle"), new("nav_active", false),
        new("battery_soc", 0.7d), new("gps_valid", true),
        new("dtc", new List<object?>()), new("warning", new List<object?>()),
    };

    private readonly string[] _slotIds;
    private readonly string[] _relationLabels;
    private readonly int _maxLen;
    private readonly int _headMaxLen;
    /// <summary>tokenizer.json `truncation.max_length`; Python applies it on every encode.</summary>
    private readonly int _truncationMax;
    private readonly JsonDocument _schema;
    private readonly Tokenizer _tokenizer;
    private readonly long _padId, _clsId, _sepId, _maskId;
    private readonly string _maskToken;
    private readonly InferenceSession _session;
    private readonly List<(int[] Ids, int[] Markers)> _headCache;
    private readonly List<(int Left, int Right)> _pairs = new();

    private VoyagPredictor(
        JsonDocument schema, Tokenizer tokenizer, InferenceSession session,
        long padId, long clsId, long sepId, long maskId, string maskToken,
        int truncationMax)
    {
        _schema = schema;
        _tokenizer = tokenizer;
        _session = session;
        _padId = padId;
        _clsId = clsId;
        _sepId = sepId;
        _maskId = maskId;
        _maskToken = maskToken;
        _truncationMax = truncationMax;

        var slots = _schema.RootElement.GetProperty("slots");
        _slotIds = slots.EnumerateObject().Select(p => p.Name).ToArray();
        _relationLabels = _schema.RootElement.GetProperty("relation_labels")
            .EnumerateArray().Select(e => e.GetString()!).ToArray();
        _maxLen = _schema.RootElement.GetProperty("model_input").GetProperty("max_len").GetInt32();
        _headMaxLen = _schema.RootElement.GetProperty("model_input").GetProperty("head_max_len").GetInt32();

        for (var i = 0; i < _slotIds.Length; i++)
        {
            for (var j = i + 1; j < _slotIds.Length; j++)
            {
                _pairs.Add((i, j));
            }
        }

        // Slot question heads are identical for every request; encode once.
        _headCache = _slotIds
            .Select(qid => BuildHead(slots.GetProperty(qid)))
            .ToList();
    }

    public static VoyagPredictor Create(string modelDir)
    {
        var schemaPath = Path.Combine(modelDir, "schema", "voyag_schema.json");
        var tokenizerPath = Path.Combine(modelDir, "tokenizer", "tokenizer.json");
        var specialPath = Path.Combine(modelDir, "tokenizer", "special_tokens_map.json");
        var onnxPath = Path.Combine(modelDir, "onnx", "voyag-v8.onnx");

        var schema = JsonDocument.Parse(File.ReadAllText(schemaPath));
        // Tokenizers.DotNet's Encode applies the post-processor (adds <bos>/<eos>);
        // Python uses add_special_tokens=False. Strip the post-processor once.
        var stripped = TokenizerCompat.EnsureNoPostProcessor(tokenizerPath);
        var tokenizer = new Tokenizer(vocabPath: stripped);

        var ids = SpecialTokenIds(tokenizerPath, specialPath);
        var truncationMax = ReadTruncationMax(tokenizerPath);
        var session = new InferenceSession(onnxPath);
        return new VoyagPredictor(schema, tokenizer, session, ids.Pad, ids.Cls, ids.Sep, ids.Mask, ids.MaskToken, truncationMax);
    }

    public string Device => _session.SessionMetadata?["ExecutionProvider"] ?? "cpu";

    private static int ReadTruncationMax(string tokenizerPath)
    {
        using var tokenizerJson = JsonDocument.Parse(File.ReadAllText(tokenizerPath));
        if (!tokenizerJson.RootElement.TryGetProperty("truncation", out var truncation)) return int.MaxValue;
        return truncation.TryGetProperty("max_length", out var maxLength) ? maxLength.GetInt32() : int.MaxValue;
    }

    private static (long Pad, long Cls, long Sep, long Mask, string MaskToken) SpecialTokenIds(
        string tokenizerPath, string specialMapPath)
    {
        using var tokenizerJson = JsonDocument.Parse(File.ReadAllText(tokenizerPath));
        using var specialMap = JsonDocument.Parse(File.ReadAllText(specialMapPath));
        var added = new Dictionary<string, long>();
        foreach (var token in tokenizerJson.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            added[token.GetProperty("content").GetString()!] = token.GetProperty("id").GetInt64();
        }
        long IdOf(string key)
        {
            var content = specialMap.RootElement.GetProperty(key).GetProperty("content").GetString()!;
            if (!added.TryGetValue(content, out var id))
            {
                throw new InvalidOperationException($"tokenizer is missing required special token: {content}");
            }
            return id;
        }
        return (IdOf("pad_token"), IdOf("cls_token"), IdOf("sep_token"), IdOf("mask_token"),
                specialMap.RootElement.GetProperty("mask_token").GetProperty("content").GetString()!);
    }

    private int[] Encode(string text, int? truncate = null)
    {
        var ids = Array.ConvertAll(_tokenizer.Encode(text), t => (int)t);
        // tokenizer.json sets truncation max_length; Python applies it on every encode.
        if (ids.Length > _truncationMax) Array.Resize(ref ids, _truncationMax);
        return truncate is null ? ids : ids.Take(truncate.Value).ToArray();
    }

    private static string[] RenderOptions(JsonElement question)
    {
        var options = new List<string>();
        foreach (var criterion in question.GetProperty("criteria").EnumerateObject())
        {
            var label = criterion.Name;
            var description = criterion.Value.GetString() ?? "";
            options.Add(description.Length == 0 ? label : $"{label}: {description}");
        }
        return options.ToArray();
    }

    private (int[] Ids, int[] Markers) BuildHead(JsonElement question)
    {
        var options = RenderOptions(question);
        var instruction = question.GetProperty("instructions").GetString()!
            .Replace(_maskToken, " ");
        var headIds = Encode($"{question.GetProperty("type").GetString()} question: {instruction}");

        var optionIds = options.Select(option =>
        {
            var tokens = Encode(" " + option.Replace(_maskToken, " "), 48);
            var ids = new int[tokens.Length + 1];
            ids[0] = (int)_maskId;
            Array.Copy(tokens, 0, ids, 1, tokens.Length);
            return ids;
        }).ToList();

        var optionBudget = _headMaxLen - optionIds.Sum(tokens => tokens.Length);
        if (optionBudget < 16)
        {
            var perOption = Math.Max(4, (_headMaxLen - 16) / Math.Max(1, optionIds.Count));
            optionIds = optionIds.Select(tokens => tokens.Take(perOption).ToArray()).ToList();
            optionBudget = _headMaxLen - optionIds.Sum(tokens => tokens.Length);
        }

        headIds = headIds.Take(Math.Max(8, optionBudget)).ToArray();
        var ids = new List<int> { (int)_clsId };
        ids.AddRange(headIds);
        ids.Add((int)_sepId);
        var markers = new List<int>();
        foreach (var tokens in optionIds)
        {
            markers.Add(ids.Count);
            ids.AddRange(tokens);
        }
        ids.Add((int)_sepId);
        return (ids.ToArray(), markers.ToArray());
    }

    /// <summary>Serialize state exactly like Python json.dumps(..., ensure_ascii=False):
    /// ", " between entries, ": " after keys, unicode kept as-is, no trailing space.</summary>
    private static string SerializeState(List<KeyValuePair<string, object?>> state)
    {
        var sb = new StringBuilder("{");
        for (var i = 0; i < state.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append('"').Append(state[i].Key).Append("\": ");
            AppendJsonValue(sb, state[i].Value);
        }
        sb.Append('}');
        return sb.ToString();
    }

    private static void AppendJsonValue(StringBuilder sb, object? value)
    {
        switch (value)
        {
            case null:
                sb.Append("null");
                break;
            case string s:
                sb.Append('"').Append(s.Replace("\\", "\\\\").Replace("\"", "\\\"")).Append('"');
                break;
            case bool b:
                sb.Append(b ? "true" : "false");
                break;
            case long or int or short or byte:
                sb.Append(Convert.ToInt64(value).ToString(CultureInfo.InvariantCulture));
                break;
            case double or float or decimal:
                sb.Append(Convert.ToDouble(value).ToString("R", CultureInfo.InvariantCulture));
                break;
            case System.Collections.IList list:
                sb.Append('[');
                for (var i = 0; i < list.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    AppendJsonValue(sb, list[i]);
                }
                sb.Append(']');
                break;
            default:
                throw new NotSupportedException($"unsupported state value type: {value?.GetType()}");
        }
    }

    private static List<KeyValuePair<string, object?>> CanonicalizeState(
        List<KeyValuePair<string, object?>> state)
    {
        if (!state.Any(p => p.Key == "utterance"))
        {
            throw new ArgumentException("state must contain 'utterance'");
        }
        var outList = new List<KeyValuePair<string, object?>>();
        foreach (var key in CanonicalStateFields)
        {
            var entry = state.FirstOrDefault(p => p.Key == key);
            if (entry.Key is not null) outList.Add(entry);
        }
        foreach (var entry in state)
        {
            if (!outList.Any(p => p.Key == entry.Key)) outList.Add(entry);
        }
        return outList;
    }

    private int[] StateIds(List<KeyValuePair<string, object?>> state)
    {
        var serialized = SerializeState(CanonicalizeState(state));
        return Encode(serialized.Replace(_maskToken, " "));
    }

    private (List<NamedOnnxValue> Feeds, int MaxMarkers) BuildInputs(
        List<KeyValuePair<string, object?>> state)
    {
        var statePart = StateIds(state);
        var rows = _headCache.Select(head =>
        {
            var room = Math.Max(0, _maxLen - head.Ids.Length - 1);
            var part = statePart.Take(room).ToArray();
            var ids = head.Ids.Concat(part).Append((int)_sepId).Take(_maxLen).ToArray();
            var markers = head.Markers.Where(m => m < _maxLen).ToArray();
            return (Ids: ids, Markers: markers);
        }).ToList();

        var maxMarkers = rows.Max(r => r.Markers.Length);
        var inputIds = new long[rows.Count * _maxLen];
        Array.Fill(inputIds, _padId);
        var attentionMask = new long[rows.Count * _maxLen];
        var markerPos = new long[rows.Count * maxMarkers];
        var markerMask = new bool[rows.Count * maxMarkers];
        var qtype = new long[rows.Count];

        for (var r = 0; r < rows.Count; r++)
        {
            for (var i = 0; i < rows[r].Ids.Length; i++)
            {
                inputIds[r * _maxLen + i] = rows[r].Ids[i];
                attentionMask[r * _maxLen + i] = 1;
            }
            for (var m = 0; m < rows[r].Markers.Length; m++)
            {
                markerPos[r * maxMarkers + m] = rows[r].Markers[m];
                markerMask[r * maxMarkers + m] = true;
            }
        }

        var feeds = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(inputIds, new[] { rows.Count, _maxLen })),
            NamedOnnxValue.CreateFromTensor("attention_mask",
                new DenseTensor<long>(attentionMask, new[] { rows.Count, _maxLen })),
            NamedOnnxValue.CreateFromTensor("marker_pos",
                new DenseTensor<long>(markerPos, new[] { rows.Count, maxMarkers })),
            NamedOnnxValue.CreateFromTensor("marker_mask",
                new DenseTensor<bool>(markerMask, new[] { rows.Count, maxMarkers })),
            NamedOnnxValue.CreateFromTensor("qtype",
                new DenseTensor<long>(qtype, new[] { rows.Count })),
        };
        return (feeds, maxMarkers);
    }

    public Dictionary<string, object?> PredictState(Dictionary<string, object?> state)
    {
        var canonical = CanonicalizeState(state.ToList());
        var (feeds, _) = BuildInputs(canonical);
        using var results = _session.Run(feeds);
        var intentProb = results.First(r => r.Name == "intent_probabilities").AsTensor<float>();
        var relationProb = results.First(r => r.Name == "relation_probabilities").AsTensor<float>();

        var active = new Dictionary<string, string>();
        var intents = new List<string[]>();
        // intent_probabilities is [num_slots, max_options]: each row is padded to
        // the widest slot with zeros. argmax over the full row, exactly like Python.
        var numCols = intentProb.Length / _slotIds.Length;
        for (var slotIndex = 0; slotIndex < _slotIds.Length; slotIndex++)
        {
            var criteria = _schema.RootElement.GetProperty("slots")
                .GetProperty(_slotIds[slotIndex]).GetProperty("criteria");
            var optionLabels = criteria.EnumerateObject().Select(p => p.Name).ToArray();
            var start = slotIndex * numCols;
            var argmax = 0;
            for (var k = 1; k < numCols; k++)
            {
                if (intentProb[start + k] > intentProb[start + argmax]) argmax = k;
            }
            if (argmax >= optionLabels.Length)
            {
                // Defensive fallback: padded columns are zero, so this never triggers.
                argmax = 0;
                for (var k = 1; k < optionLabels.Length; k++)
                {
                    if (intentProb[start + k] > intentProb[start + argmax]) argmax = k;
                }
            }
            var label = optionLabels[argmax];
            if (label == "none") continue;
            active[_slotIds[slotIndex]] = label;
            intents.Add(new[] { _slotIds[slotIndex], label });
        }

        var relations = new List<Dictionary<string, object?>>();
        var mutexViolations = new List<Dictionary<string, object?>>();
        for (var pairIndex = 0; pairIndex < _pairs.Count; pairIndex++)
        {
            var (leftIndex, rightIndex) = _pairs[pairIndex];
            var leftQid = _slotIds[leftIndex];
            var rightQid = _slotIds[rightIndex];
            if (!active.ContainsKey(leftQid) || !active.ContainsKey(rightQid)) continue;
            var start = pairIndex * 3;
            var argmax = 0;
            for (var k = 1; k < 3; k++)
            {
                if (relationProb[start + k] > relationProb[start + argmax]) argmax = k;
            }
            var relationType = _relationLabels[argmax];
            relations.Add(new Dictionary<string, object?>
            {
                ["source_id"] = leftQid,
                ["target_id"] = rightQid,
                ["type"] = relationType,
                ["confidence"] = Math.Round(relationProb[start + argmax], 4),
            });
            if (relationType == "mutually_exclusive")
            {
                mutexViolations.Add(new Dictionary<string, object?>
                {
                    ["source_id"] = leftQid,
                    ["target_id"] = rightQid,
                    ["type"] = "mutually_exclusive",
                });
            }
        }

        string executionHint;
        if (mutexViolations.Count > 0) executionHint = "clarify";
        else if (relations.Any(r => (string?)r["type"] == "sequential")) executionHint = "sequential";
        else if (relations.Any(r => (string?)r["type"] == "parallel")) executionHint = "parallel";
        else if (intents.Count > 1) executionHint = "unknown";
        else executionHint = "single";

        var utterance = canonical.FirstOrDefault(p => p.Key == "utterance").Value as string ?? "";
        return new Dictionary<string, object?>
        {
            ["utterance"] = utterance,
            ["model"] = ModelName,
            ["intents"] = intents,
            ["relations"] = relations,
            ["execution_hint"] = executionHint,
            ["mutex_violations"] = mutexViolations,
            ["clarification"] = mutexViolations.Count > 0 ? "检测到互斥意图，请确认先执行哪一个。" : null,
        };
    }

    public Dictionary<string, object?> Predict(string utterance, Dictionary<string, object?>? state = null)
    {
        var merged = new List<KeyValuePair<string, object?>>(NeutralState);
        if (state is not null)
        {
            foreach (var entry in state)
            {
                var index = merged.FindIndex(p => p.Key == entry.Key);
                if (index >= 0) merged[index] = new KeyValuePair<string, object?>(entry.Key, entry.Value);
                else merged.Add(new KeyValuePair<string, object?>(entry.Key, entry.Value));
            }
        }
        var utteranceIndex = merged.FindIndex(p => p.Key == "utterance");
        merged[utteranceIndex] = new KeyValuePair<string, object?>("utterance", utterance);
        return PredictState(new Dictionary<string, object?>(merged));
    }

    public void Dispose()
    {
        _session.Dispose();
        _tokenizer.Dispose();
        _schema.Dispose();
    }
}

/// <summary>Tokenizers.DotNet compatibility helper: strip the post-processor.</summary>
internal static class TokenizerCompat
{
    public static string EnsureNoPostProcessor(string tokenizerJsonPath)
    {
        var fullPath = Path.GetFullPath(tokenizerJsonPath);
        var cachePath = Path.Combine(Path.GetDirectoryName(fullPath)!, "tokenizer.nopost.json");

        // Strip and rewrite every time: tokenizer.json is ~10-20 MB and the pass
        // takes well under a second, which keeps the cache logic dead simple.
        using var doc = JsonDocument.Parse(File.ReadAllText(fullPath));
        using var stream = File.Create(cachePath);
        using var writer = new Utf8JsonWriter(stream);
        // JsonDocument cannot remove a member; re-serialize skipping post_processor.
        writer.WriteStartObject();
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "post_processor") continue;
            property.WriteTo(writer);
        }
        writer.WriteEndObject();
        writer.Flush();
        return cachePath;
    }
}
