#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"
work_dir="$root_dir/workdir"
cache_dir="${MODELZIP_MODEL_CACHE_DIR:-/mnt/tg/data/cache/tahoma/model-hub}"
model_ids=()
setup_model=1
setup_eval=1
langs=()

usage() {
    cat <<'EOF'
Usage: bash setup.sh [OPTIONS]

Create a Python 3.12 uv environment, install this package, and optionally fetch
baseline models and development data.

Options:
  --model-id ID              Hugging Face model ID to fetch (repeatable)
  --work-dir DIR             Work directory for models/tests (default: workdir)
 --cache-dir DIR             Base model cache directory
 --model-cache-dir DIR       Alias for --cache-dir
  --lang-pair PAIR           Development language pair to prepare (repeatable)
  --skip-model-download      Do not download/save model weights
  --skip-eval-data           Do not download development/evaluation data
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)
            model_ids+=("$2")
            shift 2
            ;;
        --work-dir)
            work_dir="$2"
            shift 2
            ;;
        --cache|--cache-dir|--model-cache-dir)
            cache_dir="$2"
            shift 2
            ;;
        --lang-pair)
            langs+=("$2")
            shift 2
            ;;
        --skip-model-download)
            setup_model=0
            shift
            ;;
        --skip-eval-data)
            setup_eval=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing it with python3 -m pip --user" >&2
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but could not be installed. See https://docs.astral.sh/uv/." >&2
    exit 1
fi

if [[ ${#model_ids[@]} -eq 0 ]]; then
    model_ids=("${MODELZIP_MODEL_ID:-google/gemma-3-12b-it}")
fi

uv venv --python 3.12 "$venv_dir"
source "$venv_dir/bin/activate"
uv pip install -e "$root_dir"

task=all
if [[ "$setup_model" == 1 && "$setup_eval" == 0 ]]; then
    task=model
elif [[ "$setup_model" == 0 && "$setup_eval" == 1 ]]; then
    task=eval
elif [[ "$setup_model" == 0 && "$setup_eval" == 0 ]]; then
    echo "Environment is ready; model and eval data setup were skipped."
    exit 0
fi

cmd=(python -m modelzip.setup --task "$task" --work "$work_dir" --cache "$cache_dir")
cmd+=(--model-id "${model_ids[@]}")
if [[ ${#langs[@]} -gt 0 ]]; then
    cmd+=(--langs "${langs[@]}")
fi

"${cmd[@]}"