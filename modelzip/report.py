#!/usr/bin/env python
#
# 2025-05-09: Initial version by TG Gowda
#
"""Leaderboard for WMT26 model-compression runs.

Combines, per language direction:
  * quality      — reference-free QE metrics read from the score cache
                   ($WORK/scores/<pair>/<out>.score.<metric>.sys), with the
                   baseline-vs-variant delta against baseline--uncompressed;
  * size         — on-disk model size and compression ratio vs the baseline;
  * speed        — median inference wall-time from the per-run stats.json
                   (throughput at the largest measured batch; batch-1 latency).

MetricX-style error metrics are lower-is-better; cometkiwi/comet are higher.
"""
import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path

from modelzip.config import WORK_DIR

PAIRS = ["ces-deu", "eng-zho_Hans", "eng-ara_EG"]
SRC_EXT = {"ces-deu": "ces", "eng-zho_Hans": "eng", "eng-ara_EG": "eng"}
DEFAULT_METRICS = ["wmt23-cometkiwi-da-xxl", "metricx-24-hybrid-xxl-v2p6-qe"]
ANCHOR = "baseline--uncompressed"
_STATS_RE = re.compile(r"\.out\.batch(\d+)\.run\d+\.stats\.json$")


def lower_is_better(metric: str) -> bool:
    m = metric.lower()
    return "metricx" in m or "error" in m


def _model_from(basename: str) -> str:
    # <test>.<src>-<tgt>.<tgt>.<model>.out.batch...
    return ".".join(basename.split(".out.batch")[0].split(".")[3:])


def short(metric: str) -> str:
    if "cometkiwi-da-xxl" in metric:
        return "ck-xxl"
    if "metricx" in metric:
        return "mx-xxl"
    return metric


def load_qe(scores_dir: Path, testset: str, metric: str) -> dict:
    out = {}
    for sc in glob.glob(str(scores_dir / "*" / f"{testset}.*.out.batch*.run*.score.{metric}.sys")):
        pair = os.path.basename(os.path.dirname(sc))
        model = _model_from(os.path.basename(sc))
        try:
            out[(model, pair)] = float(open(sc).read().strip())
        except (ValueError, OSError):
            pass
    return out


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def model_size(collected: Path, system: str):
    if not collected:
        return None
    for sub in ("model", "workdir/model"):
        p = collected / system / sub
        if p.is_dir():
            return dir_size(p)
    return None


def load_speed(tests_dir: Path, testset: str, pair: str):
    """Return (median_wall[(model,batch)], median_rss_kb[(model,batch)])."""
    walls, rss = {}, {}
    for sf in glob.glob(str(tests_dir / pair / f"{testset}.*.stats.json")):
        base = os.path.basename(sf)
        m = _STATS_RE.search(base)
        if not m:
            continue
        batch = int(m.group(1))
        model = _model_from(base)
        for line in open(sf):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("wall_time_sec") is not None:
                walls.setdefault((model, batch), []).append(float(j["wall_time_sec"]))
            if j.get("max_rss_kb") is not None:
                rss.setdefault((model, batch), []).append(float(j["max_rss_kb"]))
    med = lambda d: {k: statistics.median(v) for k, v in d.items() if v}
    return med(walls), med(rss)


def workload_chars(tests_dir: Path, testset: str, pair: str) -> int:
    src = tests_dir / pair / f"{testset}.{pair}.{SRC_EXT[pair]}"
    if not src.exists():
        return 0
    return sum(len(l.rstrip("\n")) for l in open(src, encoding="utf-8", errors="replace"))


def fmt(x, spec="{:.4f}", na="—"):
    return na if x is None else spec.format(x)


def report(work_dir: Path, collected: Path, testset: str, metrics: list):
    work_dir = Path(work_dir)
    scores_dir = work_dir / "scores"
    tests_dir = work_dir / "tests"
    qe = {m: load_qe(scores_dir, testset, m) for m in metrics}
    base_size = model_size(collected, ANCHOR)
    primary = metrics[0]

    for pair in PAIRS:
        walls, rss = load_speed(tests_dir, testset, pair)
        chars = workload_chars(tests_dir, testset, pair)
        systems = sorted({model for (model, p) in qe[primary] if p == pair})
        if not systems:
            continue
        worst = float("inf") if lower_is_better(primary) else float("-inf")
        systems.sort(key=lambda s: qe[primary].get((s, pair), worst),
                     reverse=not lower_is_better(primary))

        cols = ["#", "system"]
        for m in metrics:
            cols += [short(m), "Δ"]
        cols += ["size GB", "comp%", "thrpt ch/s", "b1 lat s", "peak GB"]
        sep = " | "
        print(f"### {pair}  ({testset}; anchor {ANCHOR})")
        print(sep.join(cols))
        print(sep.join(["---"] * len(cols)))
        for i, s in enumerate(systems, 1):
            row = [str(i), s + (" ⟵" if s == ANCHOR else "")]
            for m in metrics:
                v = qe[m].get((s, pair))
                a = qe[m].get((ANCHOR, pair))
                d = None if (v is None or a is None) else v - a
                row += [fmt(v), fmt(d, "{:+.4f}")]
            sz = model_size(collected, s)
            comp = None if (sz is None or not base_size) else 100.0 * sz / base_size
            sbatches = [b for (mm, b) in walls if mm == s]
            thr = lat = peak = None
            if sbatches:
                bmax = max(sbatches)
                w = walls.get((s, bmax))
                if w and chars:
                    thr = chars / w
                if walls.get((s, 1)):
                    lat = walls[(s, 1)]
                if rss.get((s, bmax)):
                    peak = rss[(s, bmax)] / (1024 * 1024)  # KB -> GB
            row += [fmt(sz / 1e9 if sz else None, "{:.1f}"),
                    fmt(comp, "{:.0f}"),
                    fmt(thr, "{:.0f}"),
                    fmt(lat, "{:.1f}"),
                    fmt(peak, "{:.1f}")]
            print(sep.join(row))
        print()


def main():
    parser = argparse.ArgumentParser(description="Leaderboard for WMT26 model-compression runs")
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR, help="Work dir (has tests/ and scores/)")
    parser.add_argument("-c", "--collected", type=Path,
                        default=Path.home() / "work/wmt26/model-compression/collected",
                        help="Collected submissions dir (for model sizes)")
    parser.add_argument("-t", "--testset", default="wmt26", help="Test set to report")
    parser.add_argument("-M", "--metrics", nargs="+", default=DEFAULT_METRICS, help="QE metrics (cache names)")
    args = parser.parse_args()
    report(args.work, args.collected, args.testset, args.metrics)


if __name__ == "__main__":
    main()
