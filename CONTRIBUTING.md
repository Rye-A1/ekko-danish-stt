# Contributing

Run the local checks before proposing a change:

```bash
uv sync --locked --extra local
uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'
uv build
```

Keep the Tiny native and ONNX runtimes explicit. Pin downloadable artifacts by
immutable revision and SHA-256. Add focused unit tests and real-model evidence
for a new runtime path. Do not commit model weights, datasets, credentials,
transcripts, caches, or generated build output. Preserve the separate terms for
source code and model weights.
