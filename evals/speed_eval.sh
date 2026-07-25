#!/usr/bin/env bash
#
# speed_eval.sh — measure inference speed for each collected submission on the
# ces-deu direction (the designated speed track), using the wmt26 blind set that
# is already staged. Because the set is blind (no reference), the chrf metric is
# skipped, so this times INFERENCE only. modelzip.evaluate records a stats.json
# (wall_time_sec, max_rss_kb, ...) per run, which the leaderboard reads.
#
# Default measures batch-1 single-stream latency with 3 repeats. Throughput at a
# larger batch is already available from the phase-1 QE runs (batch 16/64), but
# you can add batches here too (e.g. BATCHES="1 64").
#
# Idempotent: modelzip.evaluate skips a (batch,run) whose output already exists.
#
# Usage:
#   bash evals/speed_eval.sh                 # batch-1 latency, all systems
#   BATCHES="1 64" REPEATS=3 bash evals/speed_eval.sh
#   bash evals/speed_eval.sh <dir> [<dir>..] # only the given packages
#
# Overridable env: COLLECTED, WORK, NGPU, BATCHES, REPEATS, TESTSET, PAIR.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTED="${COLLECTED:-$HOME/work/wmt26/model-compression/collected}"
WORK="${WORK:-$HOME/work/wmt26/model-compression/eval-workdir}"
BACKUP="${BACKUP:-$WORK/backup}"
OUT="${OUT:-/tmp/speed_eval}"
NGPU="${NGPU:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"
BATCHES="${BATCHES:-1}"
REPEATS="${REPEATS:-3}"
TESTSET="${TESTSET:-wmt26}"
PAIR="${PAIR:-ces-deu}"
METRIC="${METRIC:-chrf}"                 # ref-based; skipped on the blind set (no scoring)
PER_TIMEOUT="${PER_TIMEOUT:-14400}"

PY310="${PY310:-$(uv python find 3.10 2>/dev/null || true)}"
BASE_GEMMA3="${BASE_GEMMA3:-$COLLECTED/baseline--uncompressed/workdir/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

python_bin_for() { case "$1" in fbk--*) echo "${PY310:-}" ;; *) echo "" ;; esac; }
base_model_for() { case "$1" in tmu-onono--*) [[ -d "$BASE_GEMMA3" ]] && echo "$BASE_GEMMA3" || echo "" ;; *) echo "" ;; esac; }
# ESTS gpt-oss inference.py rejects batch > 16.
cap_batch() { local id="$1" b="$2"; case "$id" in ests--*) (( b > 16 )) && echo 16 || echo "$b" ;; *) echo "$b" ;; esac; }

mkdir -p "$OUT" "$WORK"
RESULTS="$OUT/results.tsv"; : > "$RESULTS"

echo "[speed] preparing '$TESTSET' data in $WORK" >&2
python -m modelzip.setup -w "$WORK" --tests "$TESTSET" >/dev/null 2>&1 || true

run_one() {
    local dir="$1" gpu="$2"
    local id; id="$(basename "$dir")"
    local pybin; pybin="$(python_bin_for "$id")"
    local basedir; basedir="$(base_model_for "$id")"
    local log="$OUT/$id.log"; : > "$log"
    local t0; t0=$(date +%s)
    echo "[start] gpu=$gpu $id  batches='$BATCHES' repeats=$REPEATS" >&2

    if [[ ! -x "$dir/.venv/bin/python" ]]; then
        echo "[setup] $id (.venv missing)" >&2
        MODELZIP_SOURCE="$REPO" PYTHON_BIN="$pybin" BASE_MODEL_DIR="$basedir" \
            UV_VENV_CLEAR=1 timeout 1800 bash "$dir/setup.sh" >"$log" 2>&1 || {
            printf '%s\t%s\t%ss\n' "$id" "SETUP_FAIL" "$(( $(date +%s) - t0 ))" >> "$RESULTS"
            echo "[done ] $id -> SETUP_FAIL" >&2; return; }
    fi

    local b st="DONE"
    for b in $BATCHES; do
        local batch; batch="$(cap_batch "$id" "$b")"
        if ! CUDA_VISIBLE_DEVICES="$gpu" BASE_MODEL_DIR="$basedir" \
             timeout "$PER_TIMEOUT" python -m modelzip.evaluate \
               -w "$WORK" -B "$BACKUP" -b "$batch" -r "$REPEATS" -M "$METRIC" \
               -m "$dir" -l "$PAIR" -t "$TESTSET" >>"$log" 2>&1; then
            st="FAIL(b$b)"
        fi
    done
    printf '%s\t%s\t%ss\n' "$id" "$st" "$(( $(date +%s) - t0 ))" >> "$RESULTS"
    echo "[done ] $id -> $st" >&2
}

if [[ $# -gt 0 ]]; then models=("$@"); else mapfile -t models < <(ls -d "$COLLECTED"/*--*/ 2>/dev/null); fi
[[ ${#models[@]} -gt 0 ]] || { echo "no packages in $COLLECTED" >&2; exit 1; }
echo "[speed] ${#models[@]} model(s) across $NGPU GPU(s); pair=$PAIR test=$TESTSET results=$RESULTS" >&2

free=(); for ((g=0; g<NGPU; g++)); do free+=("$g"); done
declare -A slot_pid=()
reap() { local g; for g in "${!slot_pid[@]}"; do kill -0 "${slot_pid[$g]}" 2>/dev/null || { free+=("$g"); unset 'slot_pid[$g]'; }; done; }
for dir in "${models[@]}"; do
    while [[ ${#free[@]} -eq 0 ]]; do wait -n; reap; done
    gpu="${free[0]}"; free=("${free[@]:1}")
    run_one "$dir" "$gpu" &
    slot_pid[$gpu]=$!
done
wait
echo "[speed] all done. per-run timings in $WORK/tests/$PAIR/*.stats.json ; status $RESULTS" >&2
