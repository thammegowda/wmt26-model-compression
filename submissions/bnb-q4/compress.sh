#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$root_dir/.venv/bin/python" "$root_dir/prepare_model.py" \
    --model-id "${MODEL_ID:-google/gemma-3-12b-it}" \
    --cache-dir "${MODEL_CACHE:-MODEL_CACHE not set}" \
    --output "${MODEL_DIR:-$root_dir/workdir/model}" \
    --quantization bnb-q4