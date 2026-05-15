#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -d "$root_dir/.venv" ]]; then
    source "$root_dir/.venv/bin/activate"
fi

export MODELZIP_MODEL_DIR="${MODELZIP_MODEL_DIR:-$root_dir/workdir/models/gemma-3-12b-it-base}"
exec python -m modelzip.baseline "$@"