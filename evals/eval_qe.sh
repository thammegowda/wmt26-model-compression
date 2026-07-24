#!/usr/bin/env bash
#
# eval_qe.sh — phase-1 quality check. Run each collected package once at a fixed
# batch size on a real test set (default wmt25) for the directions it supports,
# store the translations in the work dir, and score them with pymarian-eval
# (wmt22-cometkiwi-da, reference-free QE). Use this to confirm quality is
# reasonable BEFORE the expensive batch-1 / repeated speed runs.
#
# Outputs land in  $WORK/tests/<pair>/  (translations + *.score); back them up
# later (e.g. azcopy the work dir). One submission per GPU.
#
# Usage:
#   bash evals/eval_qe.sh                  # all packages in COLLECTED
#   bash evals/eval_qe.sh <dir> [<dir>..]  # only the given package dirs
#
# Overridable env: COLLECTED, WORK, BACKUP, OUT, NGPU, BATCH, TESTSET, METRICS.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTED="${COLLECTED:-$HOME/work/wmt26/model-compression/collected}"
WORK="${WORK:-$HOME/work/wmt26/model-compression/eval-workdir}"
BACKUP="${BACKUP:-$WORK/backup}"
OUT="${OUT:-/tmp/eval_qe}"
NGPU="${NGPU:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"
BATCH="${BATCH:-64}"
TESTSET="${TESTSET:-wmt25}"
METRICS="${METRICS:-wmt22-cometkiwi-da}"          # QE (reference-free)
PER_TIMEOUT="${PER_TIMEOUT:-14400}"               # per submission (all its pairs), seconds

# pymarian: score env + auto-install wheel if the CLI is missing.
export PYMARIAN_CACHE="${PYMARIAN_CACHE:-/mnt/tg/data/cache/marian/metric}"
PYMARIAN_WHEEL="${PYMARIAN_WHEEL:-/mnt/tg/data/bins/marian/260623-1.12.47/pymarian-1.12.47-cp312-cp312-linux_x86_64.whl}"

# per-model env (same fixes as the sanity harness)
PY310="${PY310:-$(uv python find 3.10 2>/dev/null || true)}"
BASE_GEMMA3="${BASE_GEMMA3:-$COLLECTED/baseline--uncompressed/workdir/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# Directions each model supports (space-separated; canonical tokens).
pairs_for() {
    local all="ces-deu eng-zho_Hans eng-ara_EG"
    case "$1" in
        *_ces-deu)                                         echo "ces-deu" ;;
        *_eng-ara)                                         echo "eng-ara_EG" ;;
        *_eng-zho)                                         echo "eng-zho_Hans" ;;
        alonso--*|arc-ilsp--int4|slicers--*|*gptoss-zho-*) echo "eng-zho_Hans" ;;
        arc-ilsp--en-ar-mbr|*gptoss-arz-*)                 echo "eng-ara_EG" ;;
        arc-ilsp--vocaball-int4)                           echo "ces-deu eng-ara_EG" ;;
        fbk--*|tildeopen--*|tiny-titans--*|tmu-onono--*)   echo "ces-deu" ;;
        *)                                                 echo "$all" ;;
    esac
}
python_bin_for() { case "$1" in fbk--*) echo "${PY310:-}" ;; *) echo "" ;; esac; }
base_model_for() {
    case "$1" in
        tmu-onono--*) [[ -d "$BASE_GEMMA3" ]] && echo "$BASE_GEMMA3" || echo "" ;;
        *)            echo "" ;;
    esac
}

# --- pymarian auto-install -------------------------------------------------
if ! command -v pymarian-eval >/dev/null 2>&1; then
    echo "[eval_qe] pymarian-eval missing; installing $PYMARIAN_WHEEL" >&2
    if [[ -f "$PYMARIAN_WHEEL" ]]; then
        pip install "$PYMARIAN_WHEEL"
    else
        echo "[eval_qe] ERROR: wheel not found: $PYMARIAN_WHEEL" >&2
        exit 1
    fi
fi

mkdir -p "$OUT" "$WORK"
RESULTS="$OUT/results.tsv"
: > "$RESULTS"

# --- prepare test data once ------------------------------------------------
echo "[eval_qe] preparing '$TESTSET' data in $WORK" >&2
python -m modelzip.setup -w "$WORK" --tests "$TESTSET"

run_one() {
    local dir="$1" gpu="$2"
    local id; id="$(basename "$dir")"
    local pairs; pairs="$(pairs_for "$id")"
    local pybin; pybin="$(python_bin_for "$id")"
    local basedir; basedir="$(base_model_for "$id")"
    local log="$OUT/$id.log"
    local t0; t0=$(date +%s)
    echo "[start] gpu=$gpu batch=$BATCH pairs='$pairs' $id" >&2

    # Ensure the submission venv exists (built during sanity); rebuild if missing.
    if [[ ! -x "$dir/.venv/bin/python" ]]; then
        echo "[setup] $id (.venv missing)" >&2
        if ! MODELZIP_SOURCE="$REPO" PYTHON_BIN="$pybin" BASE_MODEL_DIR="$basedir" \
             UV_VENV_CLEAR=1 timeout 1800 bash "$dir/setup.sh" >"$log" 2>&1; then
            printf '%s\t%s\t%s\t%ss\n' "$id" "SETUP_FAIL" "$pairs" "$(( $(date +%s) - t0 ))" >> "$RESULTS"
            echo "[done ] $id -> SETUP_FAIL" >&2
            return
        fi
    fi

    # Inference (batch N) + QE scoring across the supported pairs.
    local st="DONE"
    # shellcheck disable=SC2086
    if CUDA_VISIBLE_DEVICES="$gpu" BASE_MODEL_DIR="$basedir" \
       timeout "$PER_TIMEOUT" python -m modelzip.evaluate \
         -w "$WORK" -B "$BACKUP" -b "$BATCH" -r 1 -M $METRICS \
         -m "$dir" -l $pairs -t "$TESTSET" >>"$log" 2>&1; then
        st="DONE"
    else
        local rc=$?
        [[ $rc -eq 124 ]] && st="TIMEOUT" || st="FAIL"
    fi
    # modelzip.evaluate exits 0 even if a submission's inference errored; verify
    # that an output file was produced for every requested pair.
    if [[ "$st" == "DONE" ]]; then
        local missing=0 p tgt of
        for p in $pairs; do
            tgt="${p#*-}"
            of="$WORK/tests/$p/$TESTSET.$p.$tgt.$id.out.batch$BATCH.run1"
            [[ -s "$of" ]] || missing=$((missing + 1))
        done
        [[ $missing -eq 0 ]] || st="NO_OUTPUT($missing)"
    fi
    printf '%s\t%s\t%s\t%ss\n' "$id" "$st" "$pairs" "$(( $(date +%s) - t0 ))" >> "$RESULTS"
    echo "[done ] $id -> $st" >&2
}

# Model list: args override, else all packages.
if [[ $# -gt 0 ]]; then
    models=("$@")
else
    mapfile -t models < <(ls -d "$COLLECTED"/*--*/ 2>/dev/null)
fi
[[ ${#models[@]} -gt 0 ]] || { echo "no packages found in $COLLECTED" >&2; exit 1; }
echo "[eval_qe] ${#models[@]} model(s) across $NGPU GPU(s); work=$WORK results=$RESULTS" >&2

# Bounded concurrency: one job per GPU slot.
free=(); for ((g=0; g<NGPU; g++)); do free+=("$g"); done
declare -A slot_pid=()
reap() {
    local g
    for g in "${!slot_pid[@]}"; do
        kill -0 "${slot_pid[$g]}" 2>/dev/null || { free+=("$g"); unset 'slot_pid[$g]'; }
    done
}
for dir in "${models[@]}"; do
    while [[ ${#free[@]} -eq 0 ]]; do wait -n; reap; done
    gpu="${free[0]}"; free=("${free[@]:1}")
    run_one "$dir" "$gpu" &
    slot_pid[$gpu]=$!
done
wait

# --- QE score summary ------------------------------------------------------
echo "=================== QE (${METRICS}) — ${TESTSET} b${BATCH} ===================" >&2
python3 - "$WORK/tests" "$BATCH" "$METRICS" >&2 <<'PY'
import sys, glob, os
tests_dir, batch, metric = sys.argv[1], sys.argv[2], sys.argv[3].split()[0]
rows = []
for sc in glob.glob(os.path.join(tests_dir, "*", f"*.out.batch{batch}.run1.{metric}.score")):
    base = os.path.basename(sc)
    pair = os.path.basename(os.path.dirname(sc))
    # <test>.<src>-<tgt>.<tgt>.<model>.out.batchN.run1.<metric>.score
    model = ".".join(base.split(".out.batch")[0].split(".")[3:])
    try:
        val = float(open(sc).read().strip())
    except Exception:
        val = float("nan")
    rows.append((model, pair, val))
for model, pair, val in sorted(rows):
    print(f"  {model:<52} {pair:<14} {val:.4f}")
if rows:
    import statistics
    print(f"  --- {len(rows)} (model,pair) scores; mean={statistics.mean(v for *_ ,v in rows):.4f} ---")
PY
echo "results: $RESULTS ; logs: $OUT/<id>.log ; outputs: $WORK/tests/<pair>/" >&2
