#!/usr/bin/env bash

# Debug node setup; should be done automatically in amulet setup

# bash -c "$(curl -fsSL https://raw.githubusercontent.com/thammegowda/dotfiles/master/setup.bash)"

pip install --upgrade pip
pip install -e . --no-deps
pip install -r requirements.txt

export MODELZIP_MODEL_CACHE_DIR="${MODELZIP_MODEL_CACHE_DIR:-/mnt/tg/data/cache/tahoma/model-hub}"
mkdir -p "$MODELZIP_MODEL_CACHE_DIR"

ln -sf /mnt/tg/data/cache/marian ~/.cache/marian