#!/usr/bin/env bash
#
# assemble_submissions.sh
#
# Flatten + resolve WMT26 Model Compression submissions into a clean,
# self-contained layout for offline evaluation and blob archival:
#
#     collected/<team>--<variant>/
#         setup.sh  run.sh  requirements.txt  README.md  inference.py  ...
#         workdir/model/  (or model/)   <- resolved weights
#
# Two source modes:
#   B (bundled): staging dir already contains the weights -> hardlink-copy it
#                (cp -al: instant, no extra disk, leaves staging intact).
#   H (hf):      staging dir has only code -> copy code, then download the
#                participant's pre-compressed HF repo into <model-subdir>.
#   S (slicers): like H, plus patch run.sh to force offline MODEL_DIR.
#
# HF downloads run in parallel (JOBS, default 8) and use the logged-in HF token
# (needed for the one gated Google/Gemma-4 repo). Vicomtech is handled
# separately once its bulk download finishes.
#
set -uo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
STAGING="${STAGING:-$HOME/work/wmt26/model-compression/submissions}"
FINAL="${FINAL:-$HOME/work/wmt26/model-compression/collected}"
JOBS="${JOBS:-8}"

log() { printf '%s\n' "$*" >&2; }

# manifest fields:  MODE | name | src(rel to STAGING) | hf_repo | model_subdir
read -r -d '' MANIFEST <<'EOF'
B|alonso--layeraware-native-mlp-q4|alonso|-|-
B|fbk--sq-gptq|fbk-sq-gptq|-|-
B|fbk--gptq-rtn|fbk-gptq-rtn|-|-
B|tahomamt--fp8|tahomamt-fp8|-|-
B|tahomamt--int4|tahomamt-int4|-|-
B|tahomamt--int4-bok4|tahomamt-int4-bok4|-|-
B|tahomamt--fp8-bok4|tahomamt-fp8-bok4|-|-
B|tmu-onono--diba-triton_direct|tmu-onono/submissions/diba-triton_direct|-|-
B|tmu-onono--diba-cached_unpacked|tmu-onono/submissions/diba-cached_unpacked|-|-
H|tiny-titans--gptq-int4-calibrated|tiny-titans/submissions/gptq-int4-calibrated|vania-janet/gemma-3-12b-it-gptq-int4-calibrated|workdir/model
H|tiny-titans--svd-qkv512-ff1620|tiny-titans/submissions/svd-qkv512-ff1620|alegsandyr/gemma_3_12B_qkv512_ff1620|workdir/model
H|tiny-titans--svd-qkv512-ff1620-q4-g32-full|tiny-titans/submissions/svd-qkv512-ff1620-q4-g32-full|alegsandyr/gemma_3_12B_qkv512_ff1620_q4_g32_full_submission|workdir/model
H|pare4bit--int4-gptq|pare4bit/submissions/int4-gptq|fano2458/wmt26-gemma3-12b-mt-int4-gptq|workdir/model
H|pare4bit--pruned4-healed-int4|pare4bit/submissions/pruned4-healed-int4|fano2458/wmt26-gemma3-12b-mt-pruned4-healed-int4|workdir/model
H|pare4bit--unconstrained-gemma4|pare4bit/submissions/unconstrained-gemma4|google/gemma-4-12B-it-qat-w4a16-ct|workdir/model
H|arc-ilsp--int4|arc-ilsp/submissions/int4|soksof/gemma-3-12b-wmt26-int4|workdir/model
H|arc-ilsp--fp8|arc-ilsp/submissions/fp8|soksof/gemma-3-12b-wmt26-fp8|workdir/model
H|arc-ilsp--vocaball-int4|arc-ilsp/submissions/vocaball-int4|soksof/gemma-3-12b-wmt26-vocaball-int4|workdir/model
H|arc-ilsp--en-ar-mbr|arc-ilsp/submissions/en-ar-mbr|soksof/gemma-3-12b-wmt26-vocaball-int4|workdir/model
H|cometcut--awq|cometcut/submissions/awq|Angad23/gemma-3-12b-it-awq-int4-wmt26|workdir/model
H|cometcut--gptq|cometcut/submissions/gptq|Angad23/gemma-3-12b-it-gptq-int4-wmt26|workdir/model
H|cometcut--comet-mixed-precision|cometcut/submissions/comet-mixed-precision|Angad23/gemma-3-12b-it-comet-mixed-precision-wmt26|workdir/model
H|tildeopen--15B-GPTQ-nvfp4a16|tildeopen-wmt/submissions/15B-GPTQ-nvfp4a16|TildeAI/TildeOpen15B-64k-wmt26-compression-task-cs-de-fp4a16|workdir/model
H|tildeopen--15B-RTN-fp8-dyn|tildeopen-wmt/submissions/15B-RTN-fp8-dyn|TildeAI/TildeOpen15B-64k-wmt26-compression-task-cs-de-fp8-dyn|workdir/model
H|tildeopen--8B-distill|tildeopen-wmt/submissions/8B-distill|TildeAI/TildeOpen8B-64k-wmt26-compression-task-cs-de|workdir/model
H|tildeopen--8B-distill-GPTQ-nvfp4a16|tildeopen-wmt/submissions/8B-distill-GPTQ-nvfp4a16|TildeAI/TildeOpen8B-64k-wmt26-compression-task-cs-de-fp4a16|workdir/model
H|tildeopen--8B-distill-RTN-fp8-dyn|tildeopen-wmt/submissions/8B-distill-RTN-fp8-dyn|TildeAI/TildeOpen8B-64k-wmt26-compression-task-cs-de-fp8-dyn|workdir/model
H|ests--gptoss-arz-k22|ests/submissions/ests-gptoss-arz-k22|oceanusm/ests-gptoss-arz-k22|workdir/model
H|ests--gptoss-arz-k24|ests/submissions/ests-gptoss-arz-k24|oceanusm/ests-gptoss-arz-k24|workdir/model
H|ests--gptoss-arz-k26|ests/submissions/ests-gptoss-arz-k26|oceanusm/ests-gptoss-arz-k26|workdir/model
H|ests--gptoss-zho-k26|ests/submissions/ests-gptoss-zho-k26|oceanusm/ests-gptoss-zho-k26|workdir/model
H|ests--gptoss-zho-k27|ests/submissions/ests-gptoss-zho-k27|oceanusm/ests-gptoss-zho-k27|workdir/model
H|ests--gptoss-zho-k28|ests/submissions/ests-gptoss-zho-k28|oceanusm/ests-gptoss-zho-k28|workdir/model
S|slicers--navadeep-gemma3-12b|slicers|nani-nav/gemma-3-12b-final-wmt-4488|workdir/model
EOF

has_weights() {  # any real weight file (>1MB) somewhere under $1 (pipefail-safe: no pipe)
    [[ -n "$(find "$1" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.npz' -o -name '*.gguf' \) -size +1M -print -quit 2>/dev/null)" ]]
}

hf_download() {
    local repo="$1" dir="$2"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --repo-type model --local-dir "$dir"
    else
        huggingface-cli download "$repo" --repo-type model --local-dir "$dir"
    fi
}

# --- worker: resolve one H/S entry (weights) --------------------------------
if [[ "${1:-}" == "--worker" ]]; then
    IFS='|' read -r mode name src repo subdir <<< "$2"
    dest="$FINAL/$name"
    # .RESOLVED is written only after hf download fully returns + weights verified,
    # so a partial download (killed mid-way) is never mistaken for complete.
    if [[ -f "$dest/.RESOLVED" ]]; then
        log "[skip] $name (resolved)"; exit 0
    fi
    log "[code] $name  <- $src"
    mkdir -p "$dest"
    rsync -a --exclude='.git' --exclude='.venv' --exclude='workdir/model' "$STAGING/$src"/ "$dest"/
    if [[ "$mode" == "S" ]]; then
        # slicers run.sh never sets MODEL_DIR; force it for offline eval.
        grep -q 'MODEL_DIR=' "$dest/run.sh" || sed -i \
            '0,/^DIR=.*/s//&\nexport MODEL_DIR="${MODEL_DIR:-$DIR\/workdir\/model}"/' "$dest/run.sh"
    fi
    log "[hf  ] $name  <- $repo (idempotent: completes partials, skips complete files)"
    if hf_download "$repo" "$dest/$subdir" && has_weights "$dest/$subdir"; then
        rm -rf "$dest/$subdir/.cache"   # drop HF resume metadata for a clean archive
        touch "$dest/.RESOLVED"
        log "[done] $name"; exit 0
    fi
    log "[FAIL] $name (weights not resolved)"; exit 1
fi

# ===========================================================================
# Parent
# ===========================================================================
command -v rsync >/dev/null 2>&1 || { log "[err] rsync required"; exit 1; }
mkdir -p "$FINAL"
export STAGING FINAL

hf_lines="$(mktemp)"; trap 'rm -f "$hf_lines"' EXIT

# --- Bundled entries: instant hardlink-copy (sequential) --------------------
while IFS='|' read -r mode name src repo subdir; do
    [[ -z "${mode:-}" || "$mode" == \#* ]] && continue
    if [[ "$mode" == "B" ]]; then
        dest="$FINAL/$name"
        if [[ -d "$dest" ]] && has_weights "$dest"; then
            log "[skip] $name (bundled, present)"; continue
        fi
        if [[ ! -d "$STAGING/$src" ]] || ! has_weights "$STAGING/$src"; then
            log "[WARN] $name: source not ready ($src); skipping"; continue
        fi
        log "[link] $name  <- $src"
        rm -rf "$dest"
        cp -al "$STAGING/$src" "$dest"
    else
        printf '%s|%s|%s|%s|%s\n' "$mode" "$name" "$src" "$repo" "$subdir" >> "$hf_lines"
    fi
done <<< "$MANIFEST"

# --- HF entries: parallel weight resolution ---------------------------------
n_hf=$(wc -l < "$hf_lines")
log "[ok ] resolving $n_hf HF submission(s) with up to $JOBS parallel downloads"
if command -v parallel >/dev/null 2>&1 && [[ "$JOBS" -gt 1 ]]; then
    parallel --will-cite -j "$JOBS" --line-buffer bash "$SCRIPT_PATH" --worker {} :::: "$hf_lines"
else
    while IFS= read -r line; do bash "$SCRIPT_PATH" --worker "$line"; done < "$hf_lines"
fi

# --- Summary -----------------------------------------------------------------
log ""
log "==================== assemble summary ===================="
log "collected dir : $FINAL"
declare -a OK=() BAD=()
while IFS='|' read -r mode name src repo subdir; do
    [[ -z "${mode:-}" || "$mode" == \#* ]] && continue
    dest="$FINAL/$name"
    if [[ "$mode" == "B" ]]; then
        { [[ -f "$dest/run.sh" ]] && has_weights "$dest"; } && OK+=("$name") || BAD+=("$name")
    else
        { [[ -f "$dest/.RESOLVED" && -f "$dest/run.sh" ]]; } && OK+=("$name") || BAD+=("$name")
    fi
done <<< "$MANIFEST"
log "ready  (${#OK[@]}) : ${OK[*]:-none}"
log "not-ready (${#BAD[@]}) : ${BAD[*]:-none}"
log ""
log "NOTE: vicomtech variants are assembled separately after their download completes."
[[ ${#BAD[@]} -eq 0 ]] || exit 1
