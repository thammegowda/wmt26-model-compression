#!/usr/bin/env bash
#
# score_extra_qe.sh — add a heavy GPU quality-estimation metric (reference-free)
# to translations that already exist in the eval work dir, WITHOUT re-running any
# inference. Scores every existing output file for a test set (at whatever batch
# size it was produced), one file per GPU, and caches results under scores/:
#     $SCORES/<pair>/<out_basename>.score.<metric>.sys   (system-level, 1 number)
#     $SCORES/<pair>/<out_basename>.score.<metric>.seg   (segment-level, 1/line)
# A cached metric (both files present) is skipped unless --force / FORCE=1.
#
# Default metric is cometkiwi-XXL (wmt23-cometkiwi-da-xxl), whose marian
# conversion is cached under PYMARIAN_CACHE — no HF login required. Runs on GPU
# in fp16 (verified: no fused-attention NaN for this model on H100).
#
# Usage:
#   bash evals/score_extra_qe.sh                 # score all wmt26 outputs
#   bash evals/score_extra_qe.sh --force         # recompute even if cached
#   TESTSET=wmt25 bash evals/score_extra_qe.sh   # score wmt25 outputs
#   METRIC=wmt23-cometkiwi-da-xl bash evals/score_extra_qe.sh
#
# Overridable env: WORK, TESTSET, METRIC, SCORES, NGPU, PYMARIAN_CACHE, PYMARIAN_FLAGS, FORCE.
set -uo pipefail

FORCE="${FORCE:-}"
[[ "${1:-}" == "--force" || "${1:-}" == "-f" ]] && { FORCE=1; shift; }

WORK="${WORK:-$HOME/work/wmt26/model-compression/eval-workdir}"
TESTSET="${TESTSET:-wmt26}"
METRIC="${METRIC:-wmt23-cometkiwi-da-xxl}"
NGPU="${NGPU:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"
PYMARIAN_CACHE="${PYMARIAN_CACHE:-/mnt/tg/data/cache/marian/metric}"
PYMARIAN_FLAGS="${PYMARIAN_FLAGS:---fp16 --mini-batch 16}"   # segment scores (system = mean)
TESTS="$WORK/tests"
SCORES="${SCORES:-$WORK/scores}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_score_cache.sh"

# pymarian auto-install (same wheel as eval_qe.sh).
PYMARIAN_WHEEL="${PYMARIAN_WHEEL:-/mnt/tg/data/bins/marian/260623-1.12.47/pymarian-1.12.47-cp312-cp312-linux_x86_64.whl}"
if ! command -v pymarian-eval >/dev/null 2>&1; then
    echo "[score] pymarian-eval missing; installing $PYMARIAN_WHEEL" >&2
    [[ -f "$PYMARIAN_WHEEL" ]] && pip install "$PYMARIAN_WHEEL" || { echo "[score] ERROR: wheel not found" >&2; exit 1; }
fi

# canonical pair -> source-language extension
src_ext_for() { case "$1" in ces-deu) echo ces ;; eng-zho_Hans|eng-ara_EG) echo eng ;; *) echo "" ;; esac; }

# Collect (pair, src_file, out_file) work items.
mapfile -t items < <(
  for pair in ces-deu eng-zho_Hans eng-ara_EG; do
    src="$(src_ext_for "$pair")"; [[ -n "$src" ]] || continue
    srcf="$TESTS/$pair/$TESTSET.$pair.$src"
    [[ -f "$srcf" ]] || continue
    for out in "$TESTS/$pair/$TESTSET.$pair."*.out.batch*.run1; do
      [[ -f "$out" ]] || continue
      printf '%s\t%s\t%s\n' "$pair" "$srcf" "$out"
    done
  done
)
[[ ${#items[@]} -gt 0 ]] || { echo "[score] no $TESTSET outputs under $TESTS" >&2; exit 1; }
echo "[score] metric=$METRIC  items=${#items[@]}  gpus=$NGPU  scores=$SCORES  force=${FORCE:-0}" >&2

score_one() {
    local gpu="$1" srcf="$2" out="$3"
    local sys seg; sys="$(score_sys "$out")"; seg="$(score_seg "$out")"
    if score_is_cached "$out"; then echo "[cache] gpu=$gpu $(basename "$out") -> $(cat "$sys")" >&2; return; fi
    mkdir -p "$(dirname "$seg")"
    # pymarian prints one segment score per line; system score = their mean.
    if CUDA_VISIBLE_DEVICES="$gpu" pymarian-eval --cache "$PYMARIAN_CACHE" -d 0 \
         $PYMARIAN_FLAGS -m "$METRIC" -s "$srcf" -t "$out" > "$seg.tmp" 2>"$seg.marian.log"; then
        awk '{s+=$1; n++} END{if(n) printf "%.4f", s/n}' "$seg.tmp" > "$sys"
        mv "$seg.tmp" "$seg"; rm -f "$seg.marian.log"
        echo "[done] gpu=$gpu $(basename "$out") -> $(cat "$sys")" >&2
    else
        rm -f "$seg.tmp"
        echo "[FAIL] gpu=$gpu $(basename "$out") (see $seg.marian.log)" >&2
    fi
}

# Bounded concurrency: one job per GPU slot.
free=(); for ((g=0; g<NGPU; g++)); do free+=("$g"); done
declare -A slot_pid=()
reap() { local g; for g in "${!slot_pid[@]}"; do kill -0 "${slot_pid[$g]}" 2>/dev/null || { free+=("$g"); unset 'slot_pid[$g]'; }; done; }
for it in "${items[@]}"; do
    IFS=$'\t' read -r pair srcf out <<<"$it"
    while [[ ${#free[@]} -eq 0 ]]; do wait -n; reap; done
    gpu="${free[0]}"; free=("${free[@]:1}")
    score_one "$gpu" "$srcf" "$out" &
    slot_pid[$gpu]=$!
done
wait

# --- summary ---------------------------------------------------------------
echo "=================== ${METRIC} — ${TESTSET} ===================" >&2
python3 - "$SCORES" "$TESTSET" "$METRIC" >&2 <<'PY'
import sys, glob, os
scores, ts, metric = sys.argv[1], sys.argv[2], sys.argv[3]
rows=[]
for sc in glob.glob(os.path.join(scores,"*",f"{ts}.*.out.batch*.run1.score.{metric}.sys")):
    b=os.path.basename(sc); pair=os.path.basename(os.path.dirname(sc))
    model=".".join(b.split(".out.batch")[0].split(".")[3:])
    try: v=float(open(sc).read().strip())
    except: v=float("nan")
    rows.append((pair,model,v))
for pair,model,v in sorted(rows):
    print(f"  {pair:<14} {v:.4f}  {model}")
print(f"  --- {len(rows)} scores ---")
PY
