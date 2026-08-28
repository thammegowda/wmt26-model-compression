#!/usr/bin/env bash
#
# speed_bench.sh — rigorous inference-speed benchmark for the ces-deu speed track.
#
# For every collected submission it measures wall-clock time on TWO tests at each
# batch size, with repeats:
#   * warmup : a single sentence  -> captures load/initialisation time
#   * wmt26  : the full blind set -> end-to-end wall (load + inference)
# Net (steady-state) throughput is then chars / (wmt26_wall - warmup_wall), i.e.
# with the per-batch load time subtracted (see evals/speed_report.py).
#
# Batch grid is capped per system by its declared max batch and de-duplicated.
# Repeats: REPEATS_FAST for normal systems, REPEATS_SLOW for pathologically slow
# ones (batch-1 wall > SLOW_SEC in an earlier run) to keep the tail affordable.
#
# Idempotent: modelzip.evaluate skips any (test,batch,run) whose output exists,
# so this both fills gaps in earlier runs and adds the new warmup/batch points.
#
# Usage:
#   bash evals/speed_bench.sh                    # all systems, full grid
#   BATCHES="1 64" REPEATS_FAST=3 bash evals/speed_bench.sh
#   bash evals/speed_bench.sh <dir> [<dir>..]    # only the given packages
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTED="${COLLECTED:-$HOME/work/wmt26/model-compression/collected}"
WORK="${WORK:-$HOME/work/wmt26/model-compression/eval-workdir}"
BACKUP="${BACKUP:-$WORK/backup}"
OUT="${OUT:-/tmp/speed_bench}"
NGPU="${NGPU:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"
BATCHES="${BATCHES:-1 16 64 256}"
TESTS="${TESTS:-warmup wmt26}"
REPEATS_FAST="${REPEATS_FAST:-5}"
REPEATS_SLOW="${REPEATS_SLOW:-3}"
SLOW_SEC="${SLOW_SEC:-900}"
PAIR="${PAIR:-ces-deu}"
METRIC="${METRIC:-chrf}"                 # ref-based; skipped on the blind set -> times inference only
PER_TIMEOUT="${PER_TIMEOUT:-18000}"      # 5 h cap per (batch,repeats) invocation

PY310="${PY310:-$(uv python find 3.10 2>/dev/null || true)}"
BASE_GEMMA3="${BASE_GEMMA3:-$COLLECTED/baseline--uncompressed/workdir/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
# Reduce CUDA fragmentation OOMs for naive-batching (HF transformers) systems at large
# batch sizes; recovered several batch-64/128/256 cells that OOM'd without it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin_for() { case "$1" in fbk--*) echo "${PY310:-}" ;; *) echo "" ;; esac; }
base_model_for() { case "$1" in tmu-onono--*) [[ -d "$BASE_GEMMA3" ]] && echo "$BASE_GEMMA3" || echo "" ;; *) echo "" ;; esac; }

# Declared max batch (from the participant responses), clamped to the grid max (256).
max_batch_for() {
    case "$1" in
        slicers--*|alonso--*|arc-ilsp--*) echo 8 ;;
        ests--*)                          echo 16 ;;
        tmu-onono--*|cometcut--*)         echo 32 ;;
        tahomamt--*bok4|pare4bit--*)      echo 64 ;;
        tiny-titans--*)                   echo 128 ;;
        *)                                echo 256 ;;   # tahomamt fp8/int4, fbk, vicomtech, tildeopen, baselines
    esac
}
# Effective, de-duplicated batch list for a system (each grid value clamped to its cap).
eff_batches() {
    local cap="$1" b out=() seen=" "
    for b in $BATCHES; do
        (( b > cap )) && b=$cap
        [[ "$seen" == *" $b "* ]] || { out+=("$b"); seen+="$b "; }
    done
    echo "${out[@]}"
}
# 3 repeats if a prior batch-1 wmt26 run was slower than SLOW_SEC, else 5.
repeats_for() {
    local id="$1" f w
    for f in "$WORK/tests/$PAIR/wmt26."*".$id.out.batch1.run"*.stats.json; do
        [[ -f "$f" ]] || continue
        w=$(python3 -c "import json;print(int(json.loads(open('$f').readline())['wall_time_sec']))" 2>/dev/null || echo 0)
        (( w > SLOW_SEC )) && { echo "$REPEATS_SLOW"; return; }
    done
    echo "$REPEATS_FAST"
}

mkdir -p "$OUT" "$WORK"
RESULTS="$OUT/results.tsv"; : > "$RESULTS"

echo "[bench] staging tests '$TESTS' for $PAIR in $WORK" >&2
python -m modelzip.setup -w "$WORK" -l "$PAIR" --tests $TESTS >/dev/null 2>&1 || \
    python -m modelzip.setup -w "$WORK" --tests $TESTS >/dev/null 2>&1 || true

run_one() {
    local dir="$1" gpu="$2"
    local id; id="$(basename "$dir")"
    local pybin; pybin="$(python_bin_for "$id")"
    local basedir; basedir="$(base_model_for "$id")"
    local cap; cap="$(max_batch_for "$id")"
    local batches; batches="$(eff_batches "$cap")"
    local reps; reps="$(repeats_for "$id")"
    local log="$OUT/$id.log"; : > "$log"
    local t0; t0=$(date +%s)
    echo "[start] gpu=$gpu $id  batches='$batches' repeats=$reps (cap=$cap)" >&2

    if [[ ! -x "$dir/.venv/bin/python" ]]; then
        echo "[setup] $id (.venv missing)" >&2
        MODELZIP_SOURCE="$REPO" PYTHON_BIN="$pybin" BASE_MODEL_DIR="$basedir" \
            UV_VENV_CLEAR=1 timeout 1800 bash "$dir/setup.sh" >"$log" 2>&1 || {
            printf '%s\t%s\t%ss\n' "$id" "SETUP_FAIL" "$(( $(date +%s) - t0 ))" >> "$RESULTS"
            echo "[done ] $id -> SETUP_FAIL" >&2; return; }
    fi

    local b st="DONE"
    for b in $batches; do
        if ! CUDA_VISIBLE_DEVICES="$gpu" BASE_MODEL_DIR="$basedir" \
             timeout "$PER_TIMEOUT" python -m modelzip.evaluate \
               -w "$WORK" -B "$BACKUP" -b "$b" -r "$reps" -M "$METRIC" \
               -m "$dir" -l "$PAIR" -t $TESTS >>"$log" 2>&1; then
            st="FAIL(b$b)"
        fi
    done
    printf '%s\t%s\t%ss\n' "$id" "$st" "$(( $(date +%s) - t0 ))" >> "$RESULTS"
    echo "[done ] $id -> $st  ($(( $(date +%s) - t0 ))s)" >&2
}

if [[ $# -gt 0 ]]; then models=("$@"); else mapfile -t models < <(ls -d "$COLLECTED"/*--*/ 2>/dev/null); fi
[[ ${#models[@]} -gt 0 ]] || { echo "no packages in $COLLECTED" >&2; exit 1; }
echo "[bench] ${#models[@]} model(s) x $NGPU GPU(s); pair=$PAIR grid='$BATCHES' tests='$TESTS' results=$RESULTS" >&2

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
echo "[bench] all done. per-run stats in $WORK/tests/$PAIR/*.stats.json ; status $RESULTS" >&2
