#!/usr/bin/env bash
set -euo pipefail

timeout_duration=${TIMEOUT_DURATION:-900}
lang_pair=${LANG_PAIR:-ces-deu}
batch_size=${BATCH_SIZE:-1}
skip_setup=${SKIP_SETUP:-0}

if [[ $# -gt 0 ]]; then
    candidates=("$@")
else
    candidates=()
    while IFS= read -r run_script; do
        candidates+=("$(dirname "$run_script")")
    done < <(find ./submissions -mindepth 2 -maxdepth 2 -name run.sh 2>/dev/null | sort)
fi

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No submissions found. Pass submission directories or add submissions/*/run.sh." >&2
    exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
input_file="$tmp_dir/input.jsonl"
cat > "$input_file" <<'JSONL'
{"doc_id":"sanity-doc-1","paragraph_id":1,"src_text":"Veta1"}
{"doc_id":"sanity-doc-1","paragraph_id":2,"src_text":"Veta2 slovo2 slovo3"}
{"doc_id":"sanity-doc-2","paragraph_id":1,"src_text":"Veta3"}
JSONL

for submission_dir in "${candidates[@]}"; do
    run_script="$submission_dir/run.sh"
    setup_script="$submission_dir/setup.sh"
    submission_id=$(basename "$(realpath "$submission_dir")")
    echo "===Sanity checking submission ID: $submission_id==="
    if [[ ! -f "$run_script" ]]; then
        echo "ERROR: run.sh not found in $submission_dir"
        continue
    fi

    echo "Submission directory: $(realpath "$submission_dir")"
    du -sh "$submission_dir"

    if [[ "$skip_setup" != 1 && -f "$setup_script" ]]; then
        echo "Running setup.sh for $submission_id"
        if ! timeout "$timeout_duration" bash "$setup_script"; then
            echo "ERROR: setup.sh failed for submission ID: $submission_id"
            continue
        fi
    fi

    output_file="$tmp_dir/$submission_id.out.jsonl"
    start_time=$(date +%s)
    if ! timeout "$timeout_duration" bash "$run_script" \
        --lang-pair "$lang_pair" \
        --batch-size "$batch_size" \
        --input "$input_file" \
        --output "$output_file"; then
        echo "ERROR: run.sh failed for submission ID: $submission_id"
        continue
    fi
    elapsed_time=$(($(date +%s) - start_time))
    echo "Command executed in $elapsed_time seconds for submission ID: $submission_id"

    if [[ ! -s "$output_file" ]]; then
        echo "ERROR: Output is empty for submission ID: $submission_id"
        continue
    fi
    if ! python3 - "$input_file" "$output_file" <<'PY'
import json
import sys

def records(path):
    with open(path, encoding="utf-8") as inp:
        return [json.loads(line) for line in inp if line.strip()]

inputs = records(sys.argv[1])
outputs = records(sys.argv[2])
if len(outputs) != len(inputs):
    raise SystemExit(f"output record count {len(outputs)} does not match input record count {len(inputs)}")
for index, (src, out) in enumerate(zip(inputs, outputs), start=1):
    for key in ("doc_id", "paragraph_id", "src_text"):
        if out.get(key) != src.get(key):
            raise SystemExit(f"record {index} {key} mismatch")
    if not isinstance(out.get("tgt_text"), str):
        raise SystemExit(f"record {index} missing string tgt_text")
PY
    then
        echo "ERROR: Output JSONL does not match the input contract"
        continue
    fi
    echo "SUCCESS: Submission ID $submission_id passed the sanity check."
done

echo "Sanity check completed. See ERROR messages above for any failures."
