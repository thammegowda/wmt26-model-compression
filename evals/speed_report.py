#!/usr/bin/env python3
"""Summarise the speed benchmark: load-subtracted throughput + outlier flagging.

Reads every ``*.stats.json`` under ``<work>/tests/<pair>/`` produced by
``modelzip.evaluate`` and aggregates by (system, batch, test).  For each cell it
reports the robust median wall time and the spread across repeats, flags runs
that deviate from the median (outliers), and combines the ``warmup`` (single
sentence = load) and ``wmt26`` (full set) tests into:

    gross throughput = src_chars / median(wmt26 wall)
    net   throughput = src_chars / (median(wmt26 wall) - median(warmup wall))

i.e. steady-state speed with the per-batch load/initialisation time subtracted.

Outliers matter: a run far from the median (cold cache, GPU contention, a stray
recompile) is flagged so it can be inspected rather than silently averaged in.
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

HOME = Path.home()
FNAME = re.compile(r"^(?P<test>[^.]+)\.(?P<pair>[a-z]+-[a-z]+)\.[a-z]+\."
                   r"(?P<model>.+?)\.out\.batch(?P<batch>\d+)\.run(?P<run>\d+)\.stats\.json$")


def src_chars(work: Path, pair: str, test: str) -> int:
    src, tgt = pair.split("-")
    f = work / "tests" / pair / f"{test}.{pair}.{src}"
    if not f.exists():
        return 0
    return sum(len(line) for line in f.read_text(encoding="utf-8").splitlines())


def load_runs(work: Path, pair: str):
    """-> {(model, batch, test): {run: wall_sec}}"""
    out = {}
    d = work / "tests" / pair
    for f in d.glob("*.stats.json"):
        m = FNAME.match(f.name)
        if not m or m["pair"] != pair:
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
            wall = float(rec["wall_time_sec"])
        except (ValueError, KeyError, IndexError):
            continue
        out.setdefault((m["model"], int(m["batch"]), m["test"]), {})[int(m["run"])] = wall
    return out


def robust(vals):
    """median, min, max, cv%, and outlier run-indices (|x-med| > max(25% med, 3*MAD))."""
    xs = sorted(vals.values())
    med = st.median(xs)
    mad = st.median([abs(x - med) for x in xs]) if len(xs) > 1 else 0.0
    thr = max(0.25 * med, 3 * mad)
    outliers = {r: v for r, v in vals.items() if abs(v - med) > thr and thr > 0}
    cv = (st.pstdev(xs) / st.mean(xs) * 100) if len(xs) > 1 and st.mean(xs) else 0.0
    return med, min(xs), max(xs), cv, outliers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-w", "--work", type=Path,
                    default=HOME / "work/wmt26/model-compression/eval-workdir")
    ap.add_argument("-p", "--pair", default="ces-deu")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results/speed_bench.wmt26.tsv")
    args = ap.parse_args()

    runs = load_runs(args.work, args.pair)
    chars = src_chars(args.work, args.pair, "wmt26")
    if not chars:
        print(f"[warn] no wmt26 source chars found under {args.work}/tests/{args.pair}")
    systems = sorted({k[0] for k in runs})
    batches = sorted({k[1] for k in runs})

    rows, flags = [], []
    for sysname in systems:
        for b in batches:
            full = runs.get((sysname, b, "wmt26"))
            warm = runs.get((sysname, b, "warmup"))
            if not full:
                continue
            fmed, fmin, fmax, fcv, fout = robust(full)
            wmed = robust(warm)[0] if warm else None
            gross = chars / fmed if fmed else None
            net = (chars / (fmed - wmed)) if (wmed is not None and fmed - wmed > 1e-6) else None
            load_frac = (wmed / fmed) if (wmed is not None and fmed) else None
            rows.append({
                "system": sysname, "batch": b, "n": len(full),
                "wall_med": fmed, "wall_min": fmin, "wall_max": fmax, "cv_pct": fcv,
                "load_med": wmed, "load_frac": load_frac,
                "gross_cps": gross, "net_cps": net,
                "n_outliers": len(fout),
                "outliers": ";".join(f"r{r}={v:.0f}s" for r, v in sorted(fout.items())),
            })
            for r, v in sorted(fout.items()):
                flags.append(f"  {sysname:36s} b{b:<3d} wmt26 run{r}: {v:.0f}s vs median {fmed:.0f}s "
                             f"({(v/fmed-1)*100:+.0f}%)")
            for r, v in (sorted(robust(warm)[4].items()) if warm else []):
                wm = robust(warm)[0]
                flags.append(f"  {sysname:36s} b{b:<3d} warmup run{r}: {v:.0f}s vs median {wm:.0f}s "
                             f"({(v/wm-1)*100:+.0f}%)")

    cols = ["system", "batch", "n", "wall_med", "wall_min", "wall_max", "cv_pct",
            "load_med", "load_frac", "gross_cps", "net_cps", "n_outliers", "outliers"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(
                "" if row[c] is None else (f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]))
                for c in cols) + "\n")

    print(f"[speed] {len(systems)} systems, batches {batches}, wmt26 src_chars={chars}")
    print(f"[speed] wrote {args.out}")
    # compact console view: net throughput per system across batches
    print("\n=== net throughput (chars/s, load-subtracted) by batch ===")
    hdr = "system".ljust(36) + "".join(f"b{b}".rjust(9) for b in batches) + "   load(s)@b1"
    print(hdr)
    for sysname in systems:
        cells = ""
        for b in batches:
            r = next((x for x in rows if x["system"] == sysname and x["batch"] == b), None)
            cells += (f"{r['net_cps']:.0f}".rjust(9) if r and r["net_cps"] else "-".rjust(9))
        l1 = next((x for x in rows if x["system"] == sysname and x["batch"] == 1), None)
        load = f"{l1['load_med']:.0f}" if l1 and l1["load_med"] is not None else "-"
        print(sysname.ljust(36) + cells + "   " + load)

    if flags:
        print(f"\n=== OUTLIER runs ({len(flags)}) — inspect these ===")
        print("\n".join(flags))
    else:
        print("\n=== no outliers flagged ===")


if __name__ == "__main__":
    main()
