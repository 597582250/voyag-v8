# Contributing

Voyag V8 is an inference-only release: training data, training code, and the
private training architecture are intentionally not included.

## Scope

Contributions should focus on:

- Fixing bugs in `inference.py` or the language bindings
- Improving documentation, examples, and developer experience
- Cross-language parity tests and CI coverage
- Packaging and release tooling

The public release does not accept training-data, training-code, or private
architecture contributions.

## Process

1. Open an issue describing the problem or proposal.
2. For small fixes, open a pull request directly with a clear description.
3. Keep changes scoped and include tests when behavior changes.
4. Run the relevant verification commands before opening the pull request.

## Verification

```bash
python -m py_compile inference.py examples/inference_demo.py
python tools/verify_ts_vs_python.py
```

For the TypeScript binding:

```bash
cd bindings/typescript
npm ci
npm run build
node dist/cli.js ../../ "打开车窗，同时把空调调到24度"
```

## License

By contributing, you agree that your contributions are licensed under the
project's Apache License 2.0.
