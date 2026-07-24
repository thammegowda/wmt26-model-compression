#!/usr/bin/env bash
#
# prepare_host.sh — provision the host so evals/run_all_sanity.sh can run every
# collected package. Idempotent; safe to re-run. Fixes discovered during eval:
#   - fbk needs a python3.10 interpreter (installed via uv, no root needed)
#   - tmu-onono uses stdlib `python -m venv` -> needs the python3.12-venv OS pkg
#   - tmu-onono (DiBA) needs the uncompressed gemma-3-12b-it base to reconstruct
#   - modelzip must be installable into 3.10/3.11 venvs (requires-python >=3.10)
#
# Overridable env: COLLECTED, BASE_GEMMA3.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTED="${COLLECTED:-$HOME/work/wmt26/model-compression/collected}"
BASE_GEMMA3="${BASE_GEMMA3:-$COLLECTED/baseline--uncompressed/workdir/model}"

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*" >&2; }
step() { printf '\n== %s ==\n' "$*"; }

rc=0

step "uv + python3.10 (for fbk)"
if ! command -v uv >/dev/null 2>&1; then
    warn "uv not found on PATH; install it (https://docs.astral.sh/uv/) then re-run"
    rc=1
else
    uv python install 3.10 >/dev/null 2>&1 || true
    PY310="$(uv python find 3.10 2>/dev/null || true)"
    if [[ -n "$PY310" && -x "$PY310" ]]; then
        ok "python3.10 -> $PY310 ($("$PY310" --version 2>&1))"
    else
        warn "could not provision python3.10 via uv"; rc=1
    fi
fi

step "python3.12-venv OS package (for tmu-onono stdlib venv)"
if python3.12 -c 'import ensurepip, venv' >/dev/null 2>&1; then
    ok "python3.12 venv module available"
else
    if command -v apt-get >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        if sudo -n apt-get install -y python3.12-venv >/dev/null 2>&1; then
            ok "installed python3.12-venv"
        else
            warn "apt install python3.12-venv failed"; rc=1
        fi
    else
        warn "python3.12-venv missing and passwordless sudo/apt unavailable; install it manually"
        rc=1
    fi
fi

step "base gemma-3-12b-it (for tmu-onono / DiBA)"
if [[ -f "$BASE_GEMMA3/config.json" ]] && \
   python3 - "$BASE_GEMMA3" <<'PY' >/dev/null 2>&1
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
mt = str(c.get("model_type", "")); arch = " ".join(c.get("architectures", []) or [])
sys.exit(0 if ("gemma3" in mt.lower() or "gemma3" in arch.lower()) else 1)
PY
then
    ok "valid gemma3 base -> $BASE_GEMMA3"
else
    warn "no valid gemma-3-12b-it base at $BASE_GEMMA3 (set BASE_GEMMA3 to one); tmu-onono will fail"
    rc=1
fi

step "modelzip requires-python (must allow >=3.10)"
req="$(grep -E '^\s*requires-python' "$REPO/pyproject.toml" 2>/dev/null || true)"
if echo "$req" | grep -qE '>=\s*3\.(9|10|11)'; then
    ok "modelzip $req"
else
    warn "modelzip $req — cometcut/fbk (3.11/3.10 venvs) need >=3.10"; rc=1
fi

step "summary"
if [[ $rc -eq 0 ]]; then
    echo "host ready — run: bash evals/run_all_sanity.sh"
else
    echo "host NOT fully ready — resolve [warn] items above, then re-run"
fi
exit $rc
