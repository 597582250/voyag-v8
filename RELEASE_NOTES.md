# Voyag V8 Release Notes

## Scope

This is the public ONNX local-inference release of Voyag V8.

Included:

- `onnx/voyag-v8.onnx`
- `onnx/voyag-v8.onnx.data`
- Tokenizer files
- Public schema
- `inference.py` and a runnable Python demo
- Documentation, license, security policy, and checksums

Not included:

- Training data
- Training, evaluation, or benchmark code
- Network architecture source code
- PyTorch checkpoint code or training architecture source
- Hosted inference service

## Highlights

- Runs locally with ONNX Runtime
- Does not require PyTorch
- Auto device selection: CUDAExecutionProvider when available, CPU fallback
  otherwise (GPU runtime pinned to `onnxruntime-gpu==1.23.2`; see README)
- Fixed 10-slot batch and 512-token sequence input
- Emits intent slots, pairwise relations, execution hints, mutex
  violations, and clarification
- Relation temperature is applied inside the graph

## Known issues

- Some held-out mutually exclusive cases are misclassified as parallel
- Media-domain slot accuracy is lower than other domains
- Sequential cues are sometimes underused

## Distribution note

`onnx/voyag-v8.onnx` is approximately 4.5 MB.
`onnx/voyag-v8.onnx.data` is approximately 1.3 GB and is distributed as a
GitHub Release asset.
