#!/usr/bin/env python3
"""Length-ratio hallucination analysis for WMT26 model-compression outputs.

For each direction and system it computes the per-segment output/source character
ratio; a segment whose ratio exceeds a threshold (default 2.0) is flagged as a
likely hallucination (runaway repetition / degeneration).  Per direction it then
builds the CLEAN subset of segments that NO evaluated system hallucinated, and
recomputes each system's QE quality (COMET-Kiwi-XXL higher-better, MetricX-XXL
lower-better) over that clean subset from the cached segment scores -- no
re-inference required.

Inputs:  $WORK/tests/<pair>/wmt26.<pair>.<src|tgt.system.out...>  and
         $WORK/scores/<pair>/<out>.score.<metric>.seg
Output:  results/hallucinations.wmt26.tsv  (+ optional char-ratio histogram)
"""
import argparse
import csv
import glob
import os
import re
import statistics
from pathlib import Path

PAIRS = ["ces-deu", "eng-zho_Hans", "eng-ara_EG"]
SRC_EXT = {"ces-deu": "ces", "eng-zho_Hans": "eng", "eng-ara_EG": "eng"}
CK = "wmt23-cometkiwi-da-xxl"
MX = "metricx-24-hybrid-xxl-v2p6-qe"
_BATCH_RE = re.compile(r"\.out\.batch(\d+)\.run\d+$")


def read_lines(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def read_scores(path):
    out = []
    for x in read_lines(path):
        try:
            out.append(float(x))
        except ValueError:
            out.append(None)
    return out


def model_of(out_base):
    # wmt26.<src>-<tgt>.<tgt>.<model>.out.batchN.runK
    return ".".join(out_base.split(".out.batch")[0].split(".")[3:])


def canonical_outputs(scores_dir, pair, testset):
    """Return {system: out_basename}: one run1 output per system that has a CK seg,
    preferring batch 16, else the largest batch."""
    cand = {}
    for seg in glob.glob(str(scores_dir / pair / f"{testset}.*.out.batch*.run1.score.{CK}.seg")):
        out_base = os.path.basename(seg).split(".score.")[0]
        model = model_of(out_base)
        m = _BATCH_RE.search(out_base)
        batch = int(m.group(1)) if m else 0
        # rank: prefer batch==16, then larger batch
        rank = (batch == 16, batch)
        if model not in cand or rank > cand[model][0]:
            cand[model] = (rank, out_base)
    return {s: v[1] for s, v in cand.items()}


def analyse_pair(work, pair, testset, thr):
    tests, scores = work / "tests", work / "scores"
    srcf = tests / pair / f"{testset}.{pair}.{SRC_EXT[pair]}"
    if not srcf.exists():
        return None
    src = read_lines(srcf)
    n = len(src)
    outs = canonical_outputs(scores, pair, testset)
    sysrec = {}
    for system, ob in outs.items():
        of = tests / pair / ob
        ckf = scores / pair / f"{ob}.score.{CK}.seg"
        mxf = scores / pair / f"{ob}.score.{MX}.seg"
        if not of.exists() or not ckf.exists():
            continue
        out = read_lines(of)
        cks = read_scores(ckf)
        mxs = read_scores(mxf) if mxf.exists() else [None] * len(out)
        m = min(n, len(out), len(cks), len(mxs))
        hall = {i for i in range(m) if len(src[i]) and len(out[i]) / len(src[i]) > thr}
        sysrec[system] = {"out": ob, "m": m, "hall": hall, "ck": cks, "mx": mxs}
    if not sysrec:
        return None
    m = min(r["m"] for r in sysrec.values())
    union = set().union(*(r["hall"] for r in sysrec.values()))
    clean = [i for i in range(m) if i not in union]
    rows = []
    for system, r in sysrec.items():
        ck_all = [r["ck"][i] for i in range(m) if r["ck"][i] is not None]
        ck_cln = [r["ck"][i] for i in clean if r["ck"][i] is not None]
        mx_all = [r["mx"][i] for i in range(m) if r["mx"][i] is not None]
        mx_cln = [r["mx"][i] for i in clean if r["mx"][i] is not None]
        n_hall = len({i for i in r["hall"] if i < m})
        rows.append({
            "pair": pair, "system": system, "n_seg": m, "n_hall": n_hall,
            "hall_pct": 100.0 * n_hall / m if m else 0.0,
            "ck": statistics.mean(ck_all) if ck_all else None,
            "ck_clean": statistics.mean(ck_cln) if ck_cln else None,
            "mx": statistics.mean(mx_all) if mx_all else None,
            "mx_clean": statistics.mean(mx_cln) if mx_cln else None,
        })
    rows.sort(key=lambda x: x["n_hall"], reverse=True)
    return {"n_seg": m, "n_clean": len(clean), "n_hall_union": len(union), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-w", "--work", type=Path,
                    default=Path.home() / "work/wmt26/model-compression/eval-workdir")
    ap.add_argument("-t", "--testset", default="wmt26")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="output/source char-ratio above which a segment is flagged (default 2.0)")
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("-o", "--out", type=Path, default=repo / "results/hallucinations.wmt26.tsv")
    ap.add_argument("--plot", type=Path, default=None, help="optional char-ratio histogram PDF")
    args = ap.parse_args()

    all_rows, summary = [], []
    for pair in PAIRS:
        res = analyse_pair(args.work, pair, args.testset, args.threshold)
        if not res:
            continue
        all_rows.extend(res["rows"])
        n_bad = sum(1 for r in res["rows"] if r["n_hall"] > 0)
        max_dck = max((r["ck_clean"] - r["ck"] for r in res["rows"]
                       if r["ck"] is not None and r["ck_clean"] is not None), default=0.0)
        summary.append((pair, res["n_seg"], res["n_clean"], res["n_hall_union"],
                        len(res["rows"]), n_bad, max_dck))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pair", "system", "n_seg", "n_hall", "hall_pct", "ck", "ck_clean", "mx", "mx_clean"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in all_rows:
            w.writerow(["" if r[c] is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c])
                        for c in cols])
    sum_path = args.out.with_name(args.out.stem.replace(".wmt26", "") + "_summary.wmt26.tsv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pair", "n_seg", "n_clean", "n_hall_union", "n_systems", "n_with_hall", "max_dck"])
        for pair, nseg, nclean, nun, nsys, nbad, mdck in summary:
            w.writerow([pair, nseg, nclean, nun, nsys, nbad, f"{mdck:.4f}"])

    print(f"[halluc] threshold ratio > {args.threshold}  (output/source chars)")
    print(f"{'pair':<14} {'segs':>5} {'clean':>6} {'hallUnion':>9} {'systems':>7} {'w/≥1 hall':>9}")
    for pair, nseg, nclean, nun, nsys, nbad, _mdck in summary:
        print(f"{pair:<14} {nseg:>5} {nclean:>6} {nun:>9} {nsys:>7} {nbad:>9}")
    print("\nTop hallucinators (n_hall > 0):")
    for r in sorted(all_rows, key=lambda x: x["n_hall"], reverse=True):
        if r["n_hall"] == 0:
            continue
        d_ck = (r["ck_clean"] - r["ck"]) if (r["ck"] is not None and r["ck_clean"] is not None) else None
        print(f"  {r['pair']:<14} {r['system']:<48} n_hall={r['n_hall']:>3} "
              f"ck {r['ck']:.4f}->{r['ck_clean']:.4f} (Δ{d_ck:+.4f})" if d_ck is not None
              else f"  {r['pair']:<14} {r['system']:<48} n_hall={r['n_hall']:>3}")
    print(f"\n[halluc] wrote {args.out} ({len(all_rows)} rows)")

    if args.plot:
        make_hist(args.work, args.testset, args.threshold, args.plot)


def make_hist(work, testset, thr, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(11, 3.0), sharey=True)
    for ax, pair in zip(axes, PAIRS):
        tests, scores = work / "tests", work / "scores"
        srcf = tests / pair / f"{testset}.{pair}.{SRC_EXT[pair]}"
        if not srcf.exists():
            continue
        src = read_lines(srcf)
        ratios = []
        for ob in canonical_outputs(scores, pair, testset).values():
            out = read_lines(tests / pair / ob)
            for i in range(min(len(src), len(out))):
                if len(src[i]):
                    ratios.append(len(out[i]) / len(src[i]))
        ax.hist(np.clip(ratios, 0, 3.0), bins=60, color="#2166ac", edgecolor="none")
        ax.axvline(thr, color="#b2182b", ls="--", lw=1.0)
        ax.set_title(pair, fontsize=9)
        ax.set_xlabel("output/source char ratio (clip 3)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("segments", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[halluc] wrote {outfile}")


if __name__ == "__main__":
    main()
