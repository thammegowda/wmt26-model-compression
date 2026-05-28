#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"
modelzip_source="${MODELZIP_SOURCE:-$(cd "$root_dir/../.." && pwd)}"

uv venv --python 3.12 "$venv_dir"
uv pip install --python "$venv_dir/bin/python" -r "$root_dir/requirements.txt"
uv pip install --python "$venv_dir/bin/python" --no-deps -e "$modelzip_source"