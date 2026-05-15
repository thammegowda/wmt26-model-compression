#!/usr/bin/env bash
set -euo pipefail

mydir=$(dirname "$0")
mydir=$(realpath "$mydir")
export MODELZIP_MODEL_DIR="${MODELZIP_MODEL_DIR:-$mydir}"

if [[ $# -ge 2 && "$1" != --* ]]; then
	exec python -m modelzip.baseline "$@" --model-dir "$MODELZIP_MODEL_DIR"
fi

exec python -m modelzip.baseline --model-dir "$MODELZIP_MODEL_DIR" "$@"

