#!/usr/bin/env bash
set -euo pipefail

work=$PWD/workdir
warmup_runs=3
full_runs=3
backup=/mnt/tg/data/projects/wmt26/model-compression/evals/backup-v1/${AMLT_JOB_NAME:-local}


if [[ "${1:-}" == "--baseline" ]]; then
    eval_baseline=1
    shift
else
    eval_baseline=0
fi

python -m modelzip.setup -t eval -w $work

if [[ $eval_baseline == "1" ]]; then
    # baseline + run compression
    python -m modelzip.setup -t model -w $work
    python -m modelzip.compress -m $work/models/gemma-3-12b-it-base
    models=($(ls -d $work/models/gemma-3-12b-it*))
else
    if [[ $# -gt 0 ]]; then
        models=("$@")
    else
        models=()
        [[ -f ./run.sh ]] && models+=(.)
        while IFS= read -r model_dir; do
            models+=("$model_dir")
        done < <(find ./workdir/models -mindepth 2 -maxdepth 2 -name run.sh -printf '%h\n' 2>/dev/null | sort -u)
    fi
fi

echo "Models: ${models[@]}"
echo "Backup: $backup"

metrics="chrf wmt22-comet-da wmt22-cometkiwi-da wmt23-cometkiwi-da-xl"
# get outputs on all supported lang pairs for each model so we can do quality assessment
for m in ${models[@]}; do
    echo "=====Full eval on $m with batch size 8====="
    python -m modelzip.evaluate -w $work -B $backup -r 1 -M $metrics -m $m -b 8 # -l all -t all
done

# this is for speed benchmark; try different batch sizes
for batch_size in 1 16 64 256 512; do
    for m in ${models[@]}; do
        echo "====warming $m====="
        python -m modelzip.evaluate -w $work -B $backup -r $warmup_runs -M $metrics -m $m -b 1 -l ces-deu -t warmup;

        echo "=====Speed eval for $m with batch size $batch_size on ces-deu wmt25-blind====="
        python -m modelzip.evaluate -w $work -B $backup -r $full_runs -M $metrics -m $m -b $batch_size -l ces-deu -t wmt25-blind
    done
done
