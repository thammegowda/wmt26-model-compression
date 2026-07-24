
# dry run by default, unless -y|--yes is passed
DRY_RUN=1
if [ "$1" = "-y" ] || [ "$1" = "--yes" ]; then
  DRY_RUN=0
fi

LOCAL_ROOT="$HOME/work/wmt26/model-compression"
REMOTE_ROOT="https://tgwus2.blob.core.windows.net/data/projects/wmt26/model-compression/evals"


log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $@" >&2 
}

SUB_DIRS=("collected" "submissions")
for subdir in "${SUB_DIRS[@]}"; do
    local_dir="$LOCAL_ROOT/$subdir"
    remote_dir="$REMOTE_ROOT/$subdir"
    if [ ! -d "$local_dir" ]; then
        log "SKIP (no local dir): $local_dir"
        continue
    fi
    log "Syncing  $local_dir --> $remote_dir"
    cmd="azcopy sync $local_dir $remote_dir --compare-hash=MD5 --put-md5 --local-hash-storage-mode HiddenFiles --exclude-regex '.*/\.venv/.*;.*/\.venv-compress/.*;.*/\.uv-cache/.*;.*/__pycache__/.*;.*\.RESOLVED$'"
    if [ $DRY_RUN -eq 1 ]; then
        cmd+=" --dry-run"        
    fi
    log "Executing: $cmd"
    eval "$cmd"
done

