#!/usr/bin/env bash
#
# fetch_vicomtech.sh
#
# Special-case resolver for the Vicomtech (VIC) submission.
#
# Vicomtech submitted a lightweight *stub* zip (shared code + an empty model/
# placeholder) plus a self-hosted model server that holds the actual weights,
# one directory per model variant. This script assembles each variant into a
# complete, self-contained submission:
#
#     <variant>/
#       setup.sh  run.sh  requirements.txt  README.md  inference.py   (from stub)
#       model/                                                        (downloaded)
#         config.json  model.safetensors  tokenizer.json  ...
#
# Variants download in parallel via GNU parallel (JOBS, default 12). Downloads
# land in a local staging dir (NOT the slow blobfuse mount); archive to blob
# with azcopy afterwards.
#
set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
STUB_ZIP="${STUB_ZIP:-$(dirname "$SCRIPT_PATH")/Vicomtech-stub.zip}"
DEST="${DEST:-$HOME/work/wmt26/model-compression/submissions/vicomtech}"
BASE_URL="${BASE_URL:-https://wmt26.mt.vicomtech.org}"   # note: README wget example has a typo (viomtech)
JOBS="${JOBS:-12}"

log() { printf '%s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Resolve a single variant: assemble stub + download model/. Uses env vars
# DEST, BASE_URL, TEMPLATE. Safe to run concurrently (one dir per variant).
# ---------------------------------------------------------------------------
resolve_one() {
    local variant="$1"
    local out="$DEST/$variant"
    local model_file="$out/model/model.safetensors"

    # Idempotent skip: local model matches server Content-Length.
    local remote_len local_len
    remote_len=$(wget -q -S --spider "$BASE_URL/$variant/model.safetensors" 2>&1 \
        | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
    if [[ -f "$model_file" && -n "$remote_len" ]]; then
        local_len=$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)
        if [[ "$local_len" == "$remote_len" ]]; then
            log "[skip] $variant (already complete)"
            return 0
        fi
    fi

    log "[make] $variant"
    mkdir -p "$out/model"
    # Copy shared stub code (do not clobber any already-downloaded model files).
    cp -n "$TEMPLATE"/{setup.sh,run.sh,requirements.txt,README.md,inference.py} "$out/" 2>/dev/null || true
    [[ -f "$TEMPLATE/model/README.md" ]] && cp -n "$TEMPLATE/model/README.md" "$out/model/" 2>/dev/null || true

    # -c resumes partial files (server sends no Last-Modified, so -N is useless).
    if wget -r -np -nH --cut-dirs=1 -c -nv -e robots=off \
            -R "index.html*" -P "$out/model" "$BASE_URL/$variant/"; then
        if [[ -s "$out/model/model.safetensors" && -s "$out/model/config.json" ]]; then
            log "[done] $variant"
        else
            log "[FAIL] $variant (missing model.safetensors or config.json)"
            return 1
        fi
    else
        log "[FAIL] $variant (wget error)"
        return 1
    fi
}

# Worker mode: GNU parallel invokes "bash <script> --worker <variant>".
if [[ "${1:-}" == "--worker" ]]; then
    [[ -n "${TEMPLATE:-}" && -d "${TEMPLATE:-}" ]] || { log "[err] TEMPLATE not set/invalid"; exit 1; }
    resolve_one "$2"
    exit $?
fi

command -v wget >/dev/null 2>&1 || { log "[err] wget not found"; exit 1; }
[[ -f "$STUB_ZIP" ]] || { log "[err] stub zip not found: $STUB_ZIP"; exit 1; }

# --- 1. Extract the shared stub once into a template dir ---------------------
TEMPLATE="$(mktemp -d)"
trap 'rm -rf "$TEMPLATE"' EXIT
unzip -q "$STUB_ZIP" -d "$TEMPLATE"
for f in setup.sh run.sh requirements.txt README.md inference.py; do
    [[ -e "$TEMPLATE/$f" ]] || { log "[err] stub missing expected file: $f"; exit 1; }
done
export TEMPLATE DEST BASE_URL
log "[ok ] stub extracted"

# --- 2. Determine variant list (args override; default = whole server) ------
if [[ $# -gt 0 ]]; then
    variants=("$@")
else
    mapfile -t variants < <(
        wget -q -O - "$BASE_URL/" \
        | grep -oE 'href="[^"]+/"' | sed 's/href="//; s/"//' | grep -v '\.\./' | sed 's#/$##' | sort
    )
fi
[[ ${#variants[@]} -gt 0 ]] || { log "[err] no variants discovered at $BASE_URL"; exit 1; }
mkdir -p "$DEST"
log "[ok ] ${#variants[@]} variant(s) to resolve; up to $JOBS in parallel"

# --- 3. Dispatch (GNU parallel; sequential fallback) -------------------------
if command -v parallel >/dev/null 2>&1 && [[ "$JOBS" -gt 1 ]]; then
    printf '%s\n' "${variants[@]}" \
        | parallel --will-cite -j "$JOBS" --line-buffer bash "$SCRIPT_PATH" --worker {}
else
    for v in "${variants[@]}"; do resolve_one "$v"; done
fi

# --- 4. Summary (scan disk; a variant is done if both key files exist) -------
declare -a DONE=() MISSING=()
for v in "${variants[@]}"; do
    if [[ -s "$DEST/$v/model/model.safetensors" && -s "$DEST/$v/model/config.json" ]]; then
        DONE+=("$v")
    else
        MISSING+=("$v")
    fi
done
log ""
log "==================== vicomtech summary ===================="
log "staging dir : $DEST"
log "done    (${#DONE[@]}) : ${DONE[*]:-none}"
log "missing (${#MISSING[@]}) : ${MISSING[*]:-none}"
log "total on disk: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
[[ ${#MISSING[@]} -eq 0 ]] || exit 1
