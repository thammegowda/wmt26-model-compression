#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$root_dir/.venv/bin/python" "$root_dir/inference.py" \
    --model "${MODEL_DIR:-$root_dir/workdir/model}" "$@"
