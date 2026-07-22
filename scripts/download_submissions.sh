#!/usr/bin/env bash
#
# download_submissions.sh
#
# Step 1 of collecting WMT26 Model Compression submissions.
#
# Downloads every automatable submission package (git + Hugging Face) into a
# local staging directory. Do NOT write to the blobfuse2 mount directly (it is
# slow); download here, then upload/archive to blob with azcopy manually:
#
#   local staging : ~/work/wmt26/model-compression/submissions   (DEST below)
#   blob archive  : /mnt/tg/data/projects/wmt26/model-compression/submissions
#
# Some packages contain multiple submissions (nested); flattening is a later
# step. This script only fetches the raw packages as given by the participants.
#
# A few sources (Box / Google Drive) cannot be scripted reliably and are listed
# at the end for manual download via a web browser.
#
set -uo pipefail

DEST="${DEST:-$HOME/work/wmt26/model-compression/submissions}"
mkdir -p "$DEST"

declare -a OK=()
declare -a FAILED=()
declare -a SKIPPED=()

log() { printf '%s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# git clone helper. Args: <url> <dirname> [branch]
# ---------------------------------------------------------------------------
git_clone() {
    local url="$1" dir="$2" branch="${3:-}"
    local out="$DEST/$dir"
    if [[ -d "$out/.git" ]]; then
        log "[skip] $dir (already cloned)"
        SKIPPED+=("$dir")
        return 0
    fi
    log "[git ] $url ${branch:+(branch: $branch) }-> $dir"
    if [[ -n "$branch" ]]; then
        git clone --depth 1 --branch "$branch" --single-branch "$url" "$out"
    else
        git clone --depth 1 "$url" "$out"
    fi
}

# ---------------------------------------------------------------------------
# Hugging Face download helper. Args: <repo_id> <dirname>
# Pulls the full repo (LFS weights included) into a plain local directory.
# ---------------------------------------------------------------------------
hf_download() {
    local repo="$1" dir="$2"
    local out="$DEST/$dir"
    if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
        log "[skip] $dir (already downloaded)"
        SKIPPED+=("$dir")
        return 0
    fi
    log "[hf  ] $repo -> $dir"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --repo-type model --local-dir "$out"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo" --repo-type model --local-dir "$out"
    else
        log "[err ] neither 'hf' nor 'huggingface-cli' found; pip install -U huggingface_hub"
        return 1
    fi
}

# Run a fetch, recording success/failure without aborting the whole batch.
try() {
    local label="$1"; shift
    if "$@"; then
        OK+=("$label")
    else
        log "[FAIL] $label"
        FAILED+=("$label")
    fi
}

# ===========================================================================
# GitHub repositories
#   dir name              team                 submission id(s)
# ===========================================================================
# tiny-titans            Tiny Titans          gptq-int4-calibrated,
#                                             svd-qkv512-ff1620,
#                                             svd-qkv512-ff1620-q4-g32-full
try tiny-titans      git_clone https://github.com/nicosquare/wmt26-model-compression.git tiny-titans

# slicers               SLICERS              WMT26-Submission-navadeep-gemma3-12b
try slicers          git_clone https://github.com/navadeepkiran/WMT26-Submission-navadeep-gemma3-12b.git slicers

# pare4bit              Pare4Bit             int4-gptq, unconstrained-gemma4
try pare4bit         git_clone https://github.com/fano2458/wmt26-pare4bit.git pare4bit

# arc-ilsp              arc/ilsp             vocaball-int4, int4   (branch)
try arc-ilsp         git_clone https://github.com/soksof/wmt26-model-compression.git arc-ilsp wmt26-code-submission

# cometcut              CometCut             comet-mixed-precision, gptq
try cometcut         git_clone https://github.com/angadbajwa23/wmt26-model-compression.git cometcut

# ests                  ESTS                 ests-gptoss-zho-k26, ests-gptoss-arz-k22
try ests             git_clone https://github.com/oceanusm/wmt26-ests-model-compression.git ests

# tildeopen-wmt         TildeOpen-wmt        15B-GPTQ-nvfp4a16
try tildeopen-wmt    git_clone https://github.com/tilde-nlp/wmt26-compression.git tildeopen-wmt

# ===========================================================================
# Hugging Face repositories (Link to Source Code points at the model repo)
# ===========================================================================
# alonso                alonso               layeraware-native-mlp-q4
try alonso           hf_download alonsopg/wmt26-layeraware-native-mlp-q4 alonso

# TahomaMT              4 repos across 3 rows
try tahomamt-fp8       hf_download thammegowda/wmt26zip-gemma3-12b-it-fp8       tahomamt-fp8
try tahomamt-int4      hf_download thammegowda/wmt26zip-gemma3-12b-it-int4      tahomamt-int4
try tahomamt-int4-bok4 hf_download thammegowda/wmt26zip-gemma3-12b-it-int4-bok4 tahomamt-int4-bok4
try tahomamt-fp8-bok4  hf_download thammegowda/wmt26zip-gemma3-12b-it-fp8-bok4  tahomamt-fp8-bok4

# FBK@ModelCompression  gemma3_SQ_GPTQ, gemma3-12b-GPTQ-RTN
# NOTE: FBK opted OUT of freely releasing system outputs; still collect for eval.
try fbk-sq-gptq        hf_download dsuman/gemma3-12b-SQ-GPTQ  fbk-sq-gptq
try fbk-gptq-rtn       hf_download dsuman/gemma3-12b-GPTQ-RTN fbk-gptq-rtn

# ===========================================================================
# Summary
# ===========================================================================
log ""
log "==================== download summary ===================="
log "staging dir : $DEST"
log "ok      (${#OK[@]}) : ${OK[*]:-none}"
log "skipped (${#SKIPPED[@]}) : ${SKIPPED[*]:-none}"
log "failed  (${#FAILED[@]}) : ${FAILED[*]:-none}"
log ""
log "MANUAL DOWNLOADS (Box / Google Drive - fetch via web browser into $DEST):"
log "  tmu-onono   : https://tmpuc.box.com/s/39mgvaradw2jo1930l1is3wprmwl8ttz"
log "  dragon1006  : https://drive.google.com/drive/folders/14q5Z8eXQ-mdehVvC-ks0G1tfoZiQyY5l"
log "  vic         : https://drive.google.com/drive/folders/1UmG-Curmp-2pLnPaS5ekrD5ORn9tGuEt"
log ""
log "Next: archive with azcopy, e.g."
log "  azcopy copy '$DEST/*' '<blob-sas-url>/submissions/' --recursive"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
