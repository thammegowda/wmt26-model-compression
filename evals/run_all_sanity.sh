#!/usr/bin/env bash
#
# run_all_sanity.sh — run evals/sanitycheck.sh across every collected package,
# one model per GPU, each on a translation direction the model supports.
#
# Prerequisites (see evals/prepare_host.sh):
#   - uv, a python3.10 interpreter (for fbk), python3.12-venv (for tmu-onono)
#   - a local uncompressed google/gemma-3-12b-it base (for tmu-onono / DiBA)
#   - modelzip installable into each venv (requires-python >=3.10)
#
# Usage:
#   bash evals/run_all_sanity.sh                  # all packages in COLLECTED
#   bash evals/run_all_sanity.sh <dir> [<dir>..]  # only the given package dirs
#
# Overridable env: COLLECTED, OUT, NGPU, PER_TIMEOUT, STEP_TIMEOUT, BASE_GEMMA3.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTED="${COLLECTED:-$HOME/work/wmt26/model-compression/collected}"
OUT="${OUT:-/tmp/sanity}"
NGPU="${NGPU:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"
PER_TIMEOUT="${PER_TIMEOUT:-2400}"      # hard cap per model (setup + run), seconds
STEP_TIMEOUT="${STEP_TIMEOUT:-1800}"    # sanitycheck.sh per-step (setup / run) cap

# python3.10 for fbk (uv-managed); base gemma3 for tmu-onono (DiBA reconstruct).
PY310="${PY310:-$(uv python find 3.10 2>/dev/null || true)}"
BASE_GEMMA3="${BASE_GEMMA3:-$COLLECTED/baseline--uncompressed/workdir/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"
: > "$RESULTS"

# Direction (canonical token) each model actually supports.
pair_for() {
    case "$1" in        *_ces-deu)                                              echo ces-deu ;;
        *_eng-ara)                                              echo eng-ara_EG ;;
        *_eng-zho)                                              echo eng-zho_Hans ;;        *gptoss-arz-*|*--en-ar-mbr)                          echo eng-ara_EG ;;
        *gptoss-zho-*|alonso--*|arc-ilsp--int4|slicers--*)   echo eng-zho_Hans ;;
        *)                                                   echo ces-deu ;;
    esac
}

# fbk setup.sh needs a python3.10 interpreter. Other teams read PYTHON_BIN too
# (e.g. tmu-onono), so only set it for fbk.
python_bin_for() {
    case "$1" in
        fbk--*) echo "${PY310:-}" ;;
        *)      echo "" ;;
    esac
}

# tmu-onono (DiBA) reconstructs from the uncompressed gemma-3-12b-it base; point
# it at the local copy so it neither downloads a gated model nor writes to /data.
base_model_for() {
    case "$1" in
        tmu-onono--*) [[ -d "$BASE_GEMMA3" ]] && echo "$BASE_GEMMA3" || echo "" ;;
        *)            echo "" ;;
    esac
}

run_one() {
    local dir="$1" gpu="$2"
    local id; id="$(basename "$dir")"
    local pair; pair="$(pair_for "$id")"
    local pybin; pybin="$(python_bin_for "$id")"
    local basedir; basedir="$(base_model_for "$id")"
    local log="$OUT/$id.log"
    local t0; t0=$(date +%s)
    echo "[start] gpu=$gpu pair=$pair $id" >&2
    # Start from a clean venv so setup.sh (uv or stdlib venv) is idempotent and
    # never reuses a half-built venv left by a prior failed run.
    rm -rf "$dir/.venv"
    local st
    if CUDA_VISIBLE_DEVICES="$gpu" MODELZIP_SOURCE="$REPO" \
       LANG_PAIR="$pair" TIMEOUT_DURATION="$STEP_TIMEOUT" UV_VENV_CLEAR=1 \
       PYTHON_BIN="$pybin" BASE_MODEL_DIR="$basedir" \
       timeout "$PER_TIMEOUT" bash "$REPO/evals/sanitycheck.sh" "$dir" >"$log" 2>&1; then
        if grep -q "SUCCESS: Submission ID $id passed" "$log"; then
            st=PASS
        else
            st=FAIL_VALIDATE
        fi
    else
        local rc=$?
        if [[ $rc -eq 124 ]]; then st=TIMEOUT; else st=FAIL_RUN; fi
    fi
    local dt=$(( $(date +%s) - t0 ))
    printf '%s\t%s\t%s\t%ss\n' "$id" "$st" "$pair" "$dt" >> "$RESULTS"
    echo "[done ] $id -> $st (${dt}s)" >&2
}

# Model list: args override, else all packages.
if [[ $# -gt 0 ]]; then
    models=("$@")
else
    mapfile -t models < <(ls -d "$COLLECTED"/*--*/ 2>/dev/null)
fi
[[ ${#models[@]} -gt 0 ]] || { echo "no packages found in $COLLECTED" >&2; exit 1; }
echo "scheduling ${#models[@]} model(s) across $NGPU GPU(s); results -> $RESULTS" >&2

# Bounded concurrency: one job per GPU slot.
free=(); for ((g=0; g<NGPU; g++)); do free+=("$g"); done
declare -A slot_pid=()

reap() {
    local g
    for g in "${!slot_pid[@]}"; do
        if ! kill -0 "${slot_pid[$g]}" 2>/dev/null; then
            free+=("$g"); unset 'slot_pid[$g]'
        fi
    done
}

for dir in "${models[@]}"; do
    while [[ ${#free[@]} -eq 0 ]]; do wait -n; reap; done
    gpu="${free[0]}"; free=("${free[@]:1}")
    run_one "$dir" "$gpu" &
    slot_pid[$gpu]=$!
done
wait

echo "=================== SANITY SUMMARY ===================" >&2
sort "$RESULTS" | awk -F'\t' '{printf "%-52s %-14s %-14s %s\n", $1, $2, $3, $4}' >&2
echo "-----------------------------------------------------" >&2
awk -F'\t' '{c[$2]++} END{for(k in c) printf "%-14s %d\n", k, c[k]}' "$RESULTS" >&2
echo "results: $RESULTS ; per-model logs: $OUT/<id>.log" >&2

# Non-zero exit if anything did not pass, for CI/scripting.
awk -F'\t' '$2!="PASS"{n++} END{exit (n?1:0)}' "$RESULTS"
