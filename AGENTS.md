# Ekko contributor instructions

Use `uv` for Python environments, dependencies, tests, builds, and commands.
Do not use bare `pip` or add ad hoc requirements files.

Run `uv run --no-sync ekko info` for the pinned inference manifest. Keep
model revisions, SHA-256 hashes, model terms, Hugging Face visibility, and
publication state unchanged without explicit owner approval and matching tests.
Do not read `.env` files or put credentials in commands or committed files.

The runtime wheel must not depend on browser build tools. Keep the Tiny native
and Tiny ONNX backends explicit; do not infer the runtime from a filename.
Preserve the separate licenses for code and weights.

Development checks:

```bash
uv sync --locked --extra local
uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'
uv build
```
