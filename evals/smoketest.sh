#!/usr/bin/env bash
#
# smoketest.sh — fast end-to-end sanity for a submission.
#
# Builds the submission env (setup.sh), prepares a few examples per language
# pair (smoke = WMT25 with references, smoke26 = WMT26 source-only), runs
# inference, and checks that:
#   * inference completes and writes one output line per input line, and
#   * chrF on the reference-backed `smoke` set is at least MIN_CHRF.
#
# This is intentionally small (SMOKE_N=3 examples/pair, batch 1, chrF only — no
# COMET download) so it can validate the whole setup -> inference -> score path
# in minutes before committing to a full evaluation.
#
# Usage:
#   bash evals/smoketest.sh [submission_dir ...]
#   CUDA_VISIBLE_DEVICES=0 bash evals/smoketest.sh ~/work/.../collected/baseline--uncompressed
#   # no args: discovers ./submissions/*/run.sh
#
set -uo pipefail

# Collected submissions live outside this repo, so their setup.sh can no longer
# find the modelzip package via ../..; point MODELZIP_SOURCE at the organizer repo.
export MODELZIP_SOURCE="${MODELZIP_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

work=${WORK_DIR:-$PWD/workdir}
langs=${LANGS:-"ces-deu eng-zho_Hans eng-ara_EG"}
batch_size=${BATCH_SIZE:-1}
min_chrf=${MIN_CHRF:-10}
skip_setup=${SKIP_SETUP:-0}
smoke_n=${MODELZIP_SMOKE_N:-3}
timeout_duration=${TIMEOUT_DURATION:-2400}
tests=${SMOKE_TESTS:-"smoke smoke26"}   # smoke = WMT25 (scored); smoke26 = WMT26 (source-only)
export MODELZIP_SMOKE_N="$smoke_n"

if [[ $# -gt 0 ]]; then
    submissions=("$@")
else
    submissions=()
    while IFS= read -r rs; do submissions+=("$(dirname "$rs")"); done \
        < <(find ./submissions -mindepth 2 -maxdepth 2 -name run.sh 2>/dev/null | sort)
fi
if [[ ${#submissions[@]} -eq 0 ]]; then
    echo "No submissions found. Pass submission dir(s) or add submissions/*/run.sh." >&2
    exit 1
fi

read -r -a lang_args <<< "$langs"
read -r -a test_args <<< "$tests"

# Prepare only the smoke test data (avoids the heavy wmt25-blind/ref downloads).
echo "===== preparing smoke data ($smoke_n/pair) ====="
python -m modelzip.setup -w "$work" -l "${lang_args[@]}" --tests "${test_args[@]}"

overall=0
for sub in "${submissions[@]}"; do
    sid=$(basename "$(realpath "$sub")")
    echo "==================== smoke: $sid ===================="
    if [[ ! -f "$sub/run.sh" ]]; then
        echo "FAIL[$sid]: run.sh not found in $sub"; overall=1; continue
    fi

    if [[ "$skip_setup" != 1 && -f "$sub/setup.sh" ]]; then
        echo "----- setup.sh -----"
        if ! timeout "$timeout_duration" bash "$sub/setup.sh"; then
            echo "FAIL[$sid]: setup.sh failed"; overall=1; continue
        fi
    fi

    ran_ok=1
    for t in "${test_args[@]}"; do
        echo "----- inference: $t (batch $batch_size) -----"
        if ! timeout "$timeout_duration" python -m modelzip.evaluate \
                -w "$work" -m "$sub" -t "$t" -M chrf -b "$batch_size" -B "$work/smoke-backup"; then
            echo "FAIL[$sid]: evaluate -t $t failed"; ran_ok=0; overall=1
        fi
    done
    [[ "$ran_ok" -eq 1 ]] || { echo "FAIL[$sid]"; continue; }

    # Check chrF on the reference-backed smoke set is reasonable.
    low=0; scored=0
    for pair in "${lang_args[@]}"; do
        src=${pair%%-*}; tgt=${pair#*-}
        sf=$(ls "$work/tests/$pair/smoke.$src-$tgt.$tgt.$sid.out."*.chrf.score 2>/dev/null | head -1)
        [[ -f "$sf" ]] || { echo "  $pair: (no chrF score — missing ref or run failed)"; continue; }
        score=$(tr -d '[:space:]' < "$sf")
        scored=$((scored+1))
        if awk -v s="$score" -v m="$min_chrf" 'BEGIN{exit !(s+0>=m)}'; then
            echo "  $pair: chrF=$score  OK (>= $min_chrf)"
        else
            echo "  $pair: chrF=$score  LOW (< $min_chrf)"; low=1
        fi
    done

    if [[ "$scored" -eq 0 ]]; then
        echo "WARN[$sid]: ran but produced no chrF scores"
    elif [[ "$low" -eq 0 ]]; then
        echo "PASS[$sid]"
    else
        echo "FAIL[$sid]: chrF below $min_chrf"; overall=1
    fi
done

echo "===================================================="
[[ "$overall" -eq 0 ]] && echo "SMOKE TEST: PASS" || echo "SMOKE TEST: FAIL (see above)"
exit "$overall"
