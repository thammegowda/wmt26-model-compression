# WMT26 Model Compression — Evaluator Journal

Running log of environment/packaging/runtime issues found while evaluating the
collected submissions, plus the remediation applied. Intended as source material
for the findings report (which fixes were organizer-side vs participant-side).

Environment: 8× H100 80GB, Ubuntu, system Python 3.12 only, `uv` 0.11.30, `sudo`
available. Packages restored from blob to `~/work/wmt26/model-compression/collected/`
(46 variants; `.venv` excluded from backup, rebuilt per model via each `setup.sh`).

Harness: `/tmp/run_all_sanity.sh` → `evals/sanitycheck.sh` per model, one model per
GPU (8-way), each with `MODELZIP_SOURCE=<repo>`, a supported canonical `LANG_PAIR`,
and `UV_VENV_CLEAR=1`. Sanity = setup builds venv, `run.sh` translates a 3-line
input, output must be non-empty with matching line count.

---

## Sanity run 1 (2026-07-23): 29 PASS / 17 FAIL

PASS (29): alonso; arc-ilsp (en-ar-mbr, fp8, int4, vocaball-int4); baseline
(bnb-q4, bnb-q8, uncompressed); ests (arz-k22/k24/k26, zho-k26/k27/k28); pare4bit
(int4-gptq, pruned4-healed-int4); slicers; tahomamt (fp8, fp8-bok4, int4,
int4-bok4); tildeopen (15B-GPTQ-nvfp4a16, 15B-RTN-fp8-dyn, 8B-distill,
8B-distill-GPTQ-nvfp4a16, 8B-distill-RTN-fp8-dyn); tiny-titans
(gptq-int4-calibrated, svd-qkv512-ff1620, svd-qkv512-ff1620-q4-g32-full).

### Failures by root cause

| Group | Variants | Root cause | Class | Fix |
|-------|----------|-----------|-------|-----|
| cometcut | awq, gptq, comet-mixed-precision (3) | Their `setup.sh` builds a **Python 3.11** venv, but organizer pkg `modelzip` pinned `requires-python>=3.12`, so `uv pip install -e modelzip` fails to resolve. | Organizer | Relax `modelzip` `requires-python` to `>=3.10` (code has no 3.12-only syntax; ruff already targets py310/311/312). |
| vicomtech | all 9 base variants | Raw stub `setup.sh` never creates a `.venv` (`uv pip install -r requirements.txt` with no venv); `run.sh` runs ambient `python`. Deps (incl. `vllm`) not isolated → `run.sh` fails `No module named 'vllm'`. | Assembly gap | Convert `setup.sh`/`run.sh` (all 9) to the self-contained venv pattern used by other teams. Weights live in `model/`; `inference.py --model` already defaults there. |
| fbk | gptq-rtn, sq-gptq (2) | `setup.sh` requires a **python3.10** interpreter (via `PYTHON_BIN`/`python3.10`/`conda`); host has only 3.12. | Participant pin | Provide python3.10 via `uv python install 3.10` and pass `PYTHON_BIN` to their `setup.sh`. |
| tmu-onono | diba-cached_unpacked, diba-triton_direct (2) | `setup.sh` uses stdlib `python3.12 -m venv`, but the `python3.12-venv` OS package is not installed. | Environment | `sudo apt install -y python3.12-venv`. |
| pare4bit | unconstrained-gemma4 (1) | vLLM 0.24.0 `EngineCore` dies at startup: `ImportError: cannot import name 'increment_coord' from 'cutlass.cute.core'` — vLLM ↔ `nvidia-cutlass-dsl` version conflict in the participant's pinned deps. Not OOM/model-arch. | Participant deps | Needs a compatible `nvidia-cutlass-dsl` pin (or vLLM bump). Deferred pending targeted dependency fix. |

### Notes / observations
- `uv venv` errors if a venv already exists → harness sets `UV_VENV_CLEAR=1` so
  re-runs are idempotent (first full run mis-flagged pilot models before this).
- Tokenizer warning across gemma3-derived models (`fix_mistral_regex`) — benign
  for sanity; may matter for quality scoring later.
- Vicomtech: only 9 base variants have weights; the 12 language-specialised
  variants returned HTTP 403 for `model.safetensors` (weights unavailable) and
  are not in `collected/`.

### Remediations applied (run 1 → re-run)
- [x] `modelzip` `requires-python` → `>=3.10` (cometcut ×3) — all PASS
- [x] `uv python install 3.10`; `PYTHON_BIN` passed to fbk `setup.sh` only (fbk ×2) — all PASS
- [x] `sudo apt install -y python3.12-venv` (tmu-onono ×2)
- [x] vicomtech `setup.sh`/`run.sh` → self-contained venv (×9) — all PASS
- [x] vicomtech `run.sh` → add `--gpu-memory-utilization 0.9` (×9) — all PASS
- [x] pare4bit-unconstrained-gemma4: PASS after a clean venv rebuild (the
      `cutlass.cute` import error did not recur; likely a stale/partial install).
- [x] tmu-onono: pass `BASE_MODEL_DIR=<local gemma-3-12b-it>` + writable
      `HF_HUB_CACHE` (×2) — all PASS (see below).

### Follow-on issues found during re-run
- **vicomtech (all 9), 2nd issue:** after the venv fix, vLLM loads but dies with
  `ValueError: No available memory for the cache blocks`. Their `inference.py`
  auto-computes `gpu_memory_utilization` from an over-tight VRAM estimate
  (0.094 on an 80 GB GPU → ~7.5 GB, no room for KV cache). Fix: pass
  `--gpu-memory-utilization 0.9` in `run.sh` (inference.py exposes the flag).
  Class: participant (miscalibrated auto-util on large GPUs).
- **tmu-onono (both), 2nd issue:** `setup.sh` needs the **base** `google/gemma-3-12b-it`
  to reconstruct DiBA weights; it defaults the HF cache to `/data/hf_cache/hub`
  (`mkdir /data` → Permission denied) and would otherwise download a **gated**
  model (HF not logged in). Fix: set `BASE_MODEL_DIR` to the local uncompressed
  gemma-3-12b-it (baseline--uncompressed/workdir/model), which it symlinks and
  skips the download. Class: participant (hardcoded `/data`, assumes gated base).
- **Harness robustness:** stdlib-venv teams (tmu-onono) guard venv creation with
  `[[ ! -d .venv ]]`, so a half-built `.venv` from a failed run is never rebuilt.
  Also a global `PYTHON_BIN` leaks into any team that reads it (tmu-onono).
  Fixed the runner to (a) `rm -rf .venv` before each model and (b) set
  `PYTHON_BIN` only for `fbk--*`.

### Outcome
After remediation, **46 / 46 packages PASS the sanity check** (produce a
line-count-matched translation for a supported direction). Consolidated status:
`/tmp/sanity_final.tsv`. Per-model logs retained under `/tmp/sanity*/`.

Fix ownership for the report:
- Organizer-side: `modelzip` python pin; providing python3.10 / python3.12-venv;
  harness venv-clean + per-model `PYTHON_BIN`.
- Participant-side (packaging, noted but worked around to evaluate): vicomtech
  (no isolated venv; miscalibrated gpu-mem-util); fbk (python3.10 pin);
  tmu-onono (hardcoded `/data`, gated base assumption); pare4bit-gemma4
  (transient cutlass/vLLM install).

### Reproducing the sanity run
The remediations are now scripted so the run is repeatable on any host:
- `bash evals/prepare_host.sh` — provisions python3.10 (uv), python3.12-venv
  (apt), verifies a local gemma-3-12b-it base, and checks the modelzip pin.
- `bash evals/run_all_sanity.sh` — schedules all packages one-per-GPU, choosing
  a supported direction per model and injecting the per-model env
  (`MODELZIP_SOURCE`, `PYTHON_BIN` for fbk, `BASE_MODEL_DIR` for tmu-onono,
  `UV_VENV_CLEAR`, clean `.venv`). Writes `results.tsv` + per-model logs to `OUT`
  (default `/tmp/sanity`); exits non-zero if any package does not PASS.
Verified: `run_all_sanity.sh` reproduces PASS for tmu-onono and fbk with no
manually-exported env.
