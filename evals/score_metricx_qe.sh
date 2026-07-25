#!/usr/bin/env bash
#
# score_metricx_qe.sh — score every existing translation with MetricX-24 QE (XXL)
# via `tahoma`. To avoid per-file model load/unload (~12 s/GPU each), all of a
# GPU's systems are merged into ONE stdin stream per GPU using `mtdata-map`,
# which keeps a strict 1:1 line mapping and input order and demuxes the per-line
# scores back to per-system files. Systems are sharded across GPUs and one
# single-GPU tahoma runs per shard in parallel: tahoma 0.2.0's single-process
# multi-GPU path (-d 0,1,..,7) aborts on this workload, while single-GPU is both
# stable and order-preserving (both verified).
#
# MetricX-24 is an ERROR metric: range ~0-25, LOWER is better (0 = perfect).
# System score = mean of segment scores. Written as <output>.<metric>.score,
# matching modelzip.evaluate naming so the same table tooling can read it.
#
# Batch note: paragraphs are long (tahoma truncates to 1536 tokens); -b 32 OOMs
# on 80 GB, -b 16 is safe single-GPU.
#
# Usage:
#   bash evals/score_metricx_qe.sh                 # score all wmt26 outputs
#   TESTSET=wmt25 bash evals/score_metricx_qe.sh
#   NGPU=4 BATCH=16 bash evals/score_metricx_qe.sh
#
# Overridable env: WORK, TESTSET, METRIC, TAHOMA, TAHOMA_CACHE, TAHOMA_MODEL,
#                  NGPU, BATCH.
set -uo pipefail

WORK="${WORK:-$HOME/work/wmt26/model-compression/eval-workdir}"
TESTSET="${TESTSET:-wmt26}"
METRIC="${METRIC:-metricx-24-hybrid-xxl-v2p6-qe}"
TAHOMA="${TAHOMA:-/home/aiscuser/.local/bin/tahoma/tahoma-0.2.0+2.11.0+cu130/bin/tahoma}"
TAHOMA_CACHE="${TAHOMA_CACHE:-/mnt/tg/data/cache/tahoma/model-hub}"
TAHOMA_MODEL="${TAHOMA_MODEL:-@google/metricx-24-hybrid-xxl-v2p6-bfloat16/tahoma}"
NGPU="${NGPU:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"
BATCH="${BATCH:-16}"                    # -b 32 OOMs on long paragraphs; 16 is safe single-GPU
TESTS="$WORK/tests"
STAGE="$WORK/metricx_stage/$TESTSET"

[[ -x "$TAHOMA" ]] || { echo "[metricx] tahoma not found: $TAHOMA" >&2; exit 1; }
command -v mtdata-map >/dev/null 2>&1 || { echo "[metricx] mtdata-map not on PATH" >&2; exit 1; }

src_ext_for() { case "$1" in ces-deu) echo ces ;; eng-zho_Hans|eng-ara_EG) echo eng ;; *) echo "" ;; esac; }

rm -rf "$STAGE"; mkdir -p "$STAGE"
items="$STAGE/items.tsv"; : > "$items"   # tsv <TAB> perline <TAB> out_score <TAB> N

# 1) Build one cleaned source<TAB>hyp TSV per (pair, system); skip already-scored.
for pair in ces-deu eng-zho_Hans eng-ara_EG; do
    src="$(src_ext_for "$pair")"; [[ -n "$src" ]] || continue
    srcf="$TESTS/$pair/$TESTSET.$pair.$src"; [[ -f "$srcf" ]] || continue
    N=$(wc -l < "$srcf")
    mkdir -p "$STAGE/$pair"
    for out in "$TESTS/$pair/$TESTSET.$pair."*.out.batch*.run1; do
        [[ -f "$out" ]] || continue
        sf="$out.$METRIC.score"
        [[ -s "$sf" ]] && continue
        base="$(basename "$out")"
        tsv="$STAGE/$pair/$base.tsv"
        pls="$STAGE/$pair/$base.perline"
        # align hyp to N source lines; strip tabs/CR within each field; whitespace-only
        # -> "." (mtdata-map strips trailing whitespace, so a blank field would drop the
        # tab and tahoma would abort with "needs 2 tab-separated fields").
        paste <(sed 's/\t/ /g' "$srcf") <(sed -n "1,${N}p" "$out" | sed 's/\t/ /g') \
            | awk -F'\t' 'BEGIN{OFS="\t"} {s=$1; h=$2; gsub(/\r/,"",s); gsub(/\r/,"",h);
                if(s ~ /^[[:space:]]*$/) s="."; if(h ~ /^[[:space:]]*$/) h="."; print s,h}' > "$tsv"
        Nt=$(wc -l < "$tsv")   # self-consistent count (may be source+1 due to trailing newline)
        printf '%s\t%s\t%s\t%s\n' "$tsv" "$pls" "$sf" "$Nt" >> "$items"
    done
done
n_items=$(wc -l < "$items")
[[ "$n_items" -gt 0 ]] || { echo "[metricx] nothing to score for $TESTSET (all done?)" >&2; exit 0; }
total_lines=$(awk -F'\t' '{s+=$4} END{print s}' "$items")
echo "[metricx] metric=$METRIC  systems=$n_items  segments=$total_lines  gpus=$NGPU  batch=$BATCH" >&2

# 2) Greedy (longest-processing-time) balance of systems across GPU shards by segment count.
declare -a load; for ((g=0; g<NGPU; g++)); do load[$g]=0; : > "$STAGE/listing.$g"; done
while IFS=$'\t' read -r tsv pls sf N; do
    min=0; for ((g=1; g<NGPU; g++)); do (( load[g] < load[min] )) && min=$g; done
    printf '%s\t%s\n' "$tsv" "$pls" >> "$STAGE/listing.$min"
    load[$min]=$(( load[min] + N ))
done < <(sort -t$'\t' -k4,4nr "$items")

# 3) One single-GPU tahoma per shard, all shards in parallel (one model load each).
CMD_BASE="$TAHOMA predict --cache $TAHOMA_CACHE -m $TAHOMA_MODEL --qe -d 0 -p bf16 -b $BATCH -mx 1 -i - -o -"
echo "[metricx] per-shard mapper: $CMD_BASE" >&2
t0=$(date +%s); declare -a pids=()
for ((g=0; g<NGPU; g++)); do
    [[ -s "$STAGE/listing.$g" ]] || continue
    CUDA_VISIBLE_DEVICES="$g" mtdata-map -c "$CMD_BASE" -i "$STAGE/listing.$g" > "$STAGE/shard.$g.log" 2>&1 &
    pids+=($!)
    echo "[metricx] shard $g: $(wc -l < "$STAGE/listing.$g") systems, pid $!" >&2
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "[metricx] all shards done in $(( $(date +%s) - t0 ))s (failed shards: $fail)" >&2

# 4) Reduce per-line MetricX errors to a system-level mean; write .score files.
ok=0; bad=0
while IFS=$'\t' read -r tsv pls sf N; do
    got=$(wc -l < "$pls" 2>/dev/null || echo 0)
    if [[ "$got" -eq "$N" && "$N" -gt 0 ]]; then
        awk '{s+=$1; n++} END{if(n) printf "%.4f", s/n}' "$pls" > "$sf"; ok=$((ok + 1))
    else
        echo "[metricx] WARN count mismatch ($got != $N) for $(basename "$sf")" >&2; bad=$((bad + 1))
    fi
done < "$items"
echo "[metricx] wrote $ok score files ($bad mismatched)" >&2

# 5) Summary (lower = better).
echo "=================== ${METRIC} (lower=better) — ${TESTSET} ===================" >&2
python3 - "$TESTS" "$TESTSET" "$METRIC" >&2 <<'PY'
import sys, glob, os
tests, ts, metric = sys.argv[1], sys.argv[2], sys.argv[3]
rows=[]
for sc in glob.glob(os.path.join(tests,"*",f"{ts}.*.out.batch*.run1.{metric}.score")):
    b=os.path.basename(sc); pair=os.path.basename(os.path.dirname(sc))
    model=".".join(b.split(".out.batch")[0].split(".")[3:])
    try: v=float(open(sc).read().strip())
    except: v=float("nan")
    rows.append((pair,v,model))
for pair,v,model in sorted(rows, key=lambda r:(r[0], r[1])):
    print(f"  {pair:<14} {v:7.4f}  {model}")
print(f"  --- {len(rows)} scores ---")
PY
