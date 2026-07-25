#!/usr/bin/env bash
# lib_score_cache.sh — shared metric-score cache layout for the eval work dir.
#
# Scores are cached under a scores/ root that mirrors the output-file subtree:
#
#   $SCORES/<pair>/<out_basename>.score.<metric>.sys   # system-level (one number)
#   $SCORES/<pair>/<out_basename>.score.<metric>.seg   # segment-level (one per line)
#
# where <pair>/<out_basename> is the output file's path relative to $TESTS.
# A metric is CACHED for an output iff BOTH .sys and .seg exist and are
# non-empty. Set FORCE=1 (or pass --force to the scoring scripts) to recompute.
#
# Callers must set: TESTS (output root), SCORES (cache root), METRIC, and may set FORCE.

score_sys() { local out="$1"; echo "$SCORES/${out#"$TESTS"/}.score.$METRIC.sys"; }
score_seg() { local out="$1"; echo "$SCORES/${out#"$TESTS"/}.score.$METRIC.seg"; }

# 0 = cached (skip), 1 = needs (re)compute.
score_is_cached() {
    [[ -n "${FORCE:-}" ]] && return 1
    local s g; s="$(score_sys "$1")"; g="$(score_seg "$1")"
    [[ -s "$s" && -s "$g" ]]
}
