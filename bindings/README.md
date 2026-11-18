# Voyag V8 language bindings

This directory contains local-inference bindings for Voyag V8 in languages
other than Python. Every binding reproduces the exact pipeline of the Python
runner `inference.py`:

1. load `schema/voyag_schema.json` (slots, criteria, relation labels)
2. tokenize with `tokenizer/tokenizer.json` (Metaspace BPE, 256K vocab,
   truncation max_length applied exactly like Python)
3. build the fixed `[10, 512]` input (10 slot-question heads + state JSON,
   serialized with Python `json.dumps` separators)
4. run the ONNX graph (`onnx/voyag-v8.onnx` + `voyag-v8.onnx.data`)
5. decode intents (per-row argmax over padded `[10, max_options]`),
   pairwise relations, and the execution hint

The ONNX graph and tokenizer are shared with the Python package; bindings do
not re-export or convert them.

| Binding | Directory | Tokenizer | ONNX Runtime | Status |
| --- | --- | --- | --- | --- |
| TypeScript / Node.js | `typescript/` | `@huggingface/transformers` (tokenizers.js) | `onnxruntime-node` | verified against Python CPU |
| C# / .NET 8 | `dotnet/` | `Tokenizers.DotNet` NuGet | `Microsoft.ML.OnnxRuntime` | code complete; build on a machine with the .NET SDK |
| Java 17+ | `java/` | bundled `MetaspaceBpe.java` (no native deps) | `com.microsoft.onnxruntime` | code complete; build with Maven |
| Rust | `rust/` | `tokenizers` crate (HF official) | `ort` crate | code complete; build with Cargo |

## Build & run

```text
# TypeScript / Node.js
cd bindings/typescript
npm install && npm run build
node dist/cli.js ../../ "打开车窗，同时把空调调到24度"

# .NET
cd bindings/dotnet
dotnet run --project Voyag.csproj -- ../../ "打开车窗，同时把空调调到24度"

# Java (Maven)
cd bindings/java
mvn -q compile exec:java -Dexec.args="../../ 打开车窗，同时把空调调到24度"

# Rust
cd bindings/rust
cargo build --release
./target/release/voyag-cli ../../ "打开车窗，同时把空调调到24度"
```

## Parity notes

The bindings were written against the Python runner on three subtle points
where naive ports silently drift:

- **State JSON separators**: Python serializes state with
  `json.dumps(..., ensure_ascii=False)` — `", "` between entries and `": "`
  after keys. `JSON.stringify` / `serde_json` / `System.Text.Json` emit the
  compact form, which tokenizes differently under Metaspace BPE. All bindings
  hand-serialize state to match.
- **Tokenizer truncation**: `tokenizer.json` sets `truncation.max_length: 48`,
  which Python applies on *every* encode (heads and state). tokenizers.js does
  not read this field; all bindings apply it explicitly.
- **intent_probabilities layout**: the output is `[10, max_options]` padded
  with zeros, and Python argmaxes over the full padded row. Indexing by
  per-slot option count (as a flattened layout would suggest) mis-reads rows.
- **GPU**: bindings run on CPU by default. Enable CUDA by installing the GPU
  build of each ONNX Runtime package.
- **First inference** includes kernel compilation/VRAM allocation; reuse one
  predictor instance for steady-state latency.

Cross-check script (requires Node and Python): run
`tools/verify_ts_vs_python.py` from the repository root; it compares the
TypeScript binding against the Python CPU runner on five sample utterances via
`bindings/typescript/verify_single.js`.
