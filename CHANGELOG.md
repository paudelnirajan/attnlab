# Changelog

What changed in each release, newest first. `scripts/release.sh` turns the
Unreleased section into a version heading, so write entries there as you go.
Versioning: docs/06-releasing.md.

## Unreleased

### Added
- **Self-hosting on a 16 GB Mac.** One process serves the API and the built frontend
  behind a Cloudflare Tunnel. Deploy, rollback, status and watchdog scripts, launchd
  services, and a production config (`deploy/`). Guides in docs/04–06.
- **Qwen3 0.6B (base)** in the attention and logit lens labs. It is the newest, smallest Qwen that
  TransformerLens supports, it knows 119 languages, and it uses the same tokenizer as Qwen 2.5.
- Caps on everything a request can make the server hold: stored runs by bytes, a queue limit,
  a request timeout, a whole-process memory guard, per-client rate limits, body and text limits.
- `GET /api/version`; `/api/health` reports the release, readiness, memory and counters.
- `scripts/measure_model.py` and `scripts/fetch_models.py`.

### Changed
- The server runs on CPU by default: faster than MPS for these models, and exact (D16).
- Attention runs are stored encoded (about 8× smaller) and encoded during the forward pass.
- The zoo budgets for each model's peak while loading, not only what stays resident (D17).
- Model loads work offline, including the NeelNanda checkpoints (D18).
- The frontend abandons superseded runs, re-runs expired ones, and explains proxy errors.

### Removed
- BLOOM 560M from the model zoo: it needs 8.6 GB while loading. Its tokenizer stays in the
  Tokenizer lab.
