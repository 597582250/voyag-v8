# Voyag V8 Model Card

## Model description

Voyag V8 is a structured in-vehicle intent model. It reads a natural-language
utterance together with vehicle state and produces intent slots, pairwise
relations between detected actions, and a high-level execution hint.

The public artifact is an ONNX inference graph with external weight data. The
graph is executable locally with ONNX Runtime. Training code, datasets, and
network architecture source are not included.

## Intended use

- Research on structured multi-intent vehicle understanding
- Evaluation of pairwise relation prediction
- Controlled local prototyping with ONNX Runtime

## Out-of-scope use

- Production vehicle control
- Safety-critical decision making
- Deployment without application-level validation
- Reverse-engineering the private training architecture

## Evaluation

Held-out synthetic test set:

- 5,000 groups
- 50,000 slot questions
- 4,000 relation pairs

| Head | Metric | Value |
|---|---:|---:|
| Intent | Accuracy | 0.9893 |
| Intent | Mean confidence | 0.9455 |
| Intent | ECE | 0.0526 |
| Relation | Accuracy | 0.85475 |
| Relation | Macro F1 | 0.8473 |

## Calibration

- Intent probabilities are raw choice-head softmax with identity temperature
  `1.0`. No post-hoc intent temperature was fitted.
- Relation probabilities have fitted relation temperature
  `4.628923416137695` applied inside the ONNX graph.
- Temperature scaling does not change argmax labels. It changes reported
  confidence values only.
- Intent confidence and relation confidence are not on the same scale.

## Known limitations

- Trained and evaluated only on synthetic data
- Not a safety-rated vehicle component
- Known weakness on some held-out mutually exclusive cases
- Lower media-domain slot accuracy than other domains
- Sequential cues can be underused
- Confidence is not a safety gate

## Security and safety

Voyag V8 must not be the sole decision source for safety-critical vehicle
actions. Application-level rules and validation are required. See
[SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
