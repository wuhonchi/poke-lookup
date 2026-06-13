# CardJeng Upscale Server

Self-hosted Real-ESRGAN upscale service for CardJeng. Runs on Apple Silicon
(MPS), exposed via Cloudflare Tunnel, called from cardjeng.pages.dev.

## Files

- `inference.py` — Inline RRDBNet + SRVGGNetCompact + tile-based MPS inference.
  Default model: `realesr-general-x4v3` (2.6M params, ~3 s/image on M2).
- `server.py` — FastAPI job/poll API with anonymous rate limit + size limits.
  Bypasses Cloudflare's 120 s sync request timeout.
- `bench.py` — Standalone benchmark across model × dtype × tile-size combos.
- `.gitignore` — keep weights / logs / venv out of the repo.

## Quick start

See **[M1_UPSCALE_SETUP.md](../../M1_UPSCALE_SETUP.md)** at the repo root for the
full setup walk-through (M1 Mac mini host).

## Origin

Built iteratively in a long Claude session on M2 MacBook (June 2026). See git
history for the discovery process — Codex code-review caught several real bugs
along the way (Cloudflare 120 s timeout, Swin2SR OOM, fp16 Slice kernel,
model-mismatch quality issues, canvas sizing).
