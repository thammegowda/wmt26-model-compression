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
        return "ck_xxl"
    if "metricx" in metric:
        return "mx_xxl"
    return metric.replace("-", "_").replace(".", "_")


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


CONSTRAINED_TYPES = {"gemma3", "gemma3_text"}  # the google/gemma-3-12b-it baseline family


def _config_path(collected: Path, system: str):
    for sub in ("model/config.json", "workdir/model/config.json"):
        p = collected / system / sub
        if p.is_file():
            return p
    return None


def classify(collected: Path, system: str):
    """Return (track, base_type). 'constrained' == derived from the gemma-3-12b
    baseline; 'unconstrained' == a different base model (Gemma-4, GPT-OSS,
    TildeOpen/Llama, ...), for which size/quality is not a fair compression
    comparison against our baseline."""
    if collected:
        p = _config_path(collected, system)
        if p:
            try:
                j = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                j = {}
            mt = j.get("model_type") or (j.get("text_config") or {}).get("model_type") or ""
            archs = " ".join(j.get("architectures") or [])
            if mt in CONSTRAINED_TYPES or "Gemma3" in archs:
                return "constrained", (mt or "gemma3")
            if mt or archs:
                return "unconstrained", (mt or archs)
        readme = collected / system / "README.md"
        if readme.is_file():
            txt = readme.read_text(errors="replace").lower()
            if "gemma-3" in txt or "gemma3" in txt:
                return "constrained", "gemma3"
            for o in ("gemma-4", "gemma4", "gpt-oss", "gptoss", "tildeopen", "qwen"):
                if o in txt:
                    return "unconstrained", o
    return "constrained", "unknown"


def track_of(collected: Path, system: str) -> str:
    return classify(collected, system)[0]


def build_rows(work_dir: Path, collected: Path, testset: str, metrics: list) -> list:
    """One dict per (system, pair): quality + size + speed + memory + track."""
    scores_dir = work_dir / "scores"
    tests_dir = work_dir / "tests"
    qe = {m: load_qe(scores_dir, testset, m) for m in metrics}
    base_size = model_size(collected, ANCHOR)
    meta, rows = {}, []
    for pair in PAIRS:
        walls, rss = load_speed(tests_dir, testset, pair)
        chars = workload_chars(tests_dir, testset, pair)
        systems = {model for (model, p) in qe[metrics[0]] if p == pair}
        for s in sorted(systems):
            track, base_type = meta.setdefault(s, classify(collected, s))
            sz = model_size(collected, s)
            comp = None if (sz is None or not base_size or track != "constrained") else 100.0 * sz / base_size
            thr = lat = peak = None
            sbatches = [b for (mm, b) in walls if mm == s]
            if sbatches:
                bmax = max(sbatches)
                if walls.get((s, bmax)) and chars:
                    thr = chars / walls[(s, bmax)]
                if walls.get((s, 1)):
                    lat = walls[(s, 1)]
                if rss.get((s, bmax)):
                    peak = rss[(s, bmax)] / (1024 * 1024)  # KB -> GB
            row = {"system": s, "track": track, "base_type": base_type, "pair": pair,
                   "size_bytes": sz, "size_gb": (sz / 1e9 if sz else None), "comp_pct": comp,
                   "thrpt_chars_per_s": thr, "batch1_latency_s": lat, "peak_rss_gb": peak}
            for m in metrics:
                v = qe[m].get((s, pair))
                a = qe[m].get((ANCHOR, pair))
                row[short(m)] = v
                row[short(m) + "_vs_base"] = None if (v is None or a is None) else v - a
            rows.append(row)
    return rows


def _md_row(cells):
    return "| " + " | ".join(cells) + " |"


def build_markdown(rows: list, metrics: list, testset: str) -> list:
    primary = short(metrics[0])
    lower = lower_is_better(metrics[0])
    worst = float("inf") if lower else float("-inf")
    out = [f"# WMT26 Model-Compression Leaderboard — {testset}", ""]
    out += [
        "Reference-free QE on the blind set. **Δ** = system − `baseline--uncompressed` (same direction).",
        "",
        "- **ck_xxl** = cometkiwi-XXL (higher is better) · **mx_xxl** = MetricX-24-XXL (lower is better)",
        "- **size GB** on-disk weights · **comp%** = size vs baseline (constrained only) · "
        "**thrpt ch/s** = source chars ÷ wall-time at the largest measured batch · "
        "**b1 lat s** = batch-1 single-stream wall-time on ces-deu (speed track) · **peak GB** = peak host RSS",
        "- Tracks: **constrained** = compress gemma-3-12b · **unconstrained** = different base model (shown separately)",
        "",
    ]
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    for pair in PAIRS:
        prs = by_pair.get(pair)
        if not prs:
            continue
        out += [f"## {pair}", ""]
        for track, title, show_comp in (
            ("constrained", "Constrained (compress gemma-3-12b; Δ/comp% vs baseline--uncompressed)", True),
            ("unconstrained", "Unconstrained (different base model; not a compression ratio)", False),
        ):
            sub = [r for r in prs if r["track"] == track]
            if not sub:
                continue
            sub.sort(key=lambda r: (r[primary] if r.get(primary) is not None else worst), reverse=not lower)
            cols = ["#", "system"]
            for m in metrics:
                cols += [short(m), "Δ"]
            cols += ["size GB"] + (["comp%"] if show_comp else []) + ["thrpt ch/s", "b1 lat s", "peak GB"]
            out += [f"### {pair} — {title}", "", _md_row(cols), _md_row(["---"] * len(cols))]
            for i, r in enumerate(sub, 1):
                line = [str(i), r["system"] + (" ⟵" if r["system"] == ANCHOR else "")]
                for m in metrics:
                    line += [fmt(r.get(short(m))), fmt(r.get(short(m) + "_vs_base"), "{:+.4f}")]
                line += [fmt(r.get("size_gb"), "{:.1f}")]
                if show_comp:
                    line += [fmt(r.get("comp_pct"), "{:.0f}")]
                line += [fmt(r.get("thrpt_chars_per_s"), "{:.0f}"),
                         fmt(r.get("batch1_latency_s"), "{:.1f}"),
                         fmt(r.get("peak_rss_gb"), "{:.1f}")]
                out.append(_md_row(line))
            out.append("")
    return out


def _tsv_val(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_tsv(rows: list, path: Path, metrics: list):
    """Wide: one row per (system, pair) with every metric + size/speed/memory."""
    cols = ["system", "track", "base_type", "pair"]
    for m in metrics:
        cols += [short(m), short(m) + "_vs_base"]
    cols += ["size_bytes", "size_gb", "comp_pct", "thrpt_chars_per_s", "batch1_latency_s", "peak_rss_gb"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(_tsv_val(r.get(c)) for c in cols) + "\n")


def write_long_tsv(rows: list, path: Path, metrics: list):
    """Tidy long: one row per (system, pair, metric) — handy for faceted plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(["system", "track", "base_type", "pair", "metric", "score", "vs_base"]) + "\n")
        for r in rows:
            for m in metrics:
                f.write("\t".join([r["system"], r["track"], r["base_type"], r["pair"], short(m),
                                   _tsv_val(r.get(short(m))), _tsv_val(r.get(short(m) + "_vs_base"))]) + "\n")


def report(work_dir: Path, collected: Path, testset: str, metrics: list, out_dir: Path = None):
    work_dir = Path(work_dir)
    rows = build_rows(work_dir, collected, testset, metrics)
    md = build_markdown(rows, metrics, testset)
    print("\n".join(md))
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"leaderboard.{testset}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        write_tsv(rows, out_dir / f"summary.{testset}.tsv", metrics)
        write_long_tsv(rows, out_dir / f"metrics_long.{testset}.tsv", metrics)
        print(f"[report] wrote leaderboard.{testset}.md, summary.{testset}.tsv, "
              f"metrics_long.{testset}.tsv to {out_dir} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Leaderboard for WMT26 model-compression runs")
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR, help="Work dir (has tests/ and scores/)")
    parser.add_argument("-c", "--collected", type=Path,
                        default=Path.home() / "work/wmt26/model-compression/collected",
                        help="Collected submissions dir (for model sizes)")
    parser.add_argument("-t", "--testset", default="wmt26", help="Test set to report")
    parser.add_argument("-M", "--metrics", nargs="+", default=DEFAULT_METRICS, help="QE metrics (cache names)")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("-o", "--out-dir", type=Path, default=repo_root / "results",
                        help="Directory for results/*.tsv (pass '' to skip)")
    args = parser.parse_args()
    out_dir = args.out_dir if str(args.out_dir) not in ("", ".") else None
    report(args.work, args.collected, args.testset, args.metrics, out_dir=out_dir)


if __name__ == "__main__":
    main()
