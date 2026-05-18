#!/usr/bin/env bash
set -euo pipefail

work=${WORK_DIR:-$PWD/workdir}
warmup_runs=${WARMUP_RUNS:-3}
full_runs=${FULL_RUNS:-3}
job_name=${JOB_NAME:-${SUB_ID:-local}}
backup=${BACKUP_DIR:-/mnt/tg/data/projects/wmt26/model-compression/evals/backup-v1/$job_name}
skip_setup=${SKIP_SETUP:-0}

if [[ $# -gt 0 ]]; then
    submissions=("$@")
else
    submissions=()
    while IFS= read -r run_script; do
        submissions+=("$(dirname "$run_script")")
    done < <(find ./submissions -mindepth 2 -maxdepth 2 -name run.sh 2>/dev/null | sort)
fi

if [[ ${#submissions[@]} -eq 0 ]]; then
    echo "No submissions found. Pass submission directories or add submissions/*/run.sh." >&2
    exit 1
fi

python -m modelzip.setup -w "$work"

echo "Submissions: ${submissions[*]}"
echo "Backup: $backup"

metrics=${METRICS:-"chrf wmt22-comet-da wmt22-cometkiwi-da wmt23-cometkiwi-da-xl"}

for submission in "${submissions[@]}"; do
    setup_script="$submission/setup.sh"
    if [[ "$skip_setup" != 1 && -f "$setup_script" ]]; then
        echo "=====Setup for $submission====="
        bash "$setup_script"
    fi

    echo "=====Full eval on $submission with batch size 8====="
    python -m modelzip.evaluate -w "$work" -B "$backup" -r 1 -M $metrics -m "$submission" -b 8

    for batch_size in 1 16 64 256 512; do
        echo "====Warming $submission====="
        python -m modelzip.evaluate -w "$work" -B "$backup" -r "$warmup_runs" -M $metrics -m "$submission" -b 1 -l ces-deu -t warmup

        echo "=====Speed eval for $submission with batch size $batch_size on ces-deu wmt25-blind====="
        python -m modelzip.evaluate -w "$work" -B "$backup" -r "$full_runs" -M $metrics -m "$submission" -b "$batch_size" -l ces-deu -t wmt25-blind
    done
done
