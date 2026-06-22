#!/usr/bin/env python

# 2025-05-09: Initial version by TG Gowda

"""
Evaluation pipeline. The tests and metrics are for demo purposes only.
The official evaluation will use held-out WMT26 test sets.
"""
import argparse
import json
import logging as LOG
import os
import subprocess as sp
import tempfile
from pathlib import Path

from modelzip.config import DEF_BATCH_SIZE, DEF_LANG_PAIRS, TASK_CONF, WORK_DIR, normalize_lang_pair
from modelzip.submission import read_input_records, validate_output_records
import shutil
import time, resource

DEF_SHOW_PROGRESS = False
PYMARIAN_CACHE = os.getenv("PYMARIAN_CACHE", "/mnt/tg/data/cache/marian/metric")
PYMARIAN_EXTRA = os.getenv("PYMARIAN_EXTRA", "-c 16")  # default: CPU threads (GPU fused attention NaN on H100)


def get_score(src_file: Path, out_file: Path, ref_file: Path, metric: str):
    if metric == "chrf":
        cmd = f"sacrebleu {ref_file} -i {out_file} -m {metric} -b -lc"
    else:
        cmd = f"pymarian-eval --cache {PYMARIAN_CACHE} {PYMARIAN_EXTRA} -m {metric} -r {ref_file} -t {out_file} -s {src_file} -a only"
    LOG.info(f"Scoring: {cmd}")
    return sp.check_output(cmd, shell=True, text=True).strip()


def get_run_cmd(model_dir: Path) -> list[str]:
    """Return the command for a submission directory containing run.sh."""
    run_script = model_dir / "run.sh"
    assert run_script.exists(), f"run.sh not found in {model_dir}"
    return ["bash", str(run_script)]


def _reference_text(record: dict) -> str | None:
    refs = record.get("refs") or {}
    ref = refs.get("refA")
    if isinstance(ref, dict):
        ref = ref.get("ref")
    return ref if isinstance(ref, str) else None


def _load_input_records(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as inp:
        return read_input_records(inp)


def validate_jsonl_output(input_file: Path, output_file: Path) -> None:
    with open(output_file, "r", encoding="utf-8", errors="replace") as out:
        validate_output_records(_load_input_records(input_file), out)


def write_participant_input(source_file: Path, target_file: Path) -> None:
    with open(target_file, "w", encoding="utf-8") as out:
        for record in _load_input_records(source_file):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metric_files(input_file: Path, output_file: Path, ref_file: Path, out_dir: Path) -> tuple[Path, Path, Path] | None:
    input_records = _load_input_records(input_file)
    ref_by_key = {}
    if not ref_file.exists() or ref_file.stat().st_size == 0:
        return None
    with open(ref_file, "r", encoding="utf-8", errors="replace") as refs_in:
        for record in (json.loads(line) for line in refs_in if line.strip()):
            ref_by_key[(record["doc_id"], record["paragraph_id"])] = _reference_text(record)

    with open(output_file, "r", encoding="utf-8", errors="replace") as out:
        output_records = validate_output_records(input_records, out)

    src_lines, out_lines, ref_lines = [], [], []
    for input_record, output_record in zip(input_records, output_records):
        ref = ref_by_key.get((input_record["doc_id"], input_record["paragraph_id"]))
        if ref is None:
            continue
        src_lines.append(input_record["src_text"])
        out_lines.append(output_record["tgt_text"])
        ref_lines.append(ref)
    if not ref_lines:
        return None

    src_text = out_dir / "src.txt"
    out_text = out_dir / "out.txt"
    ref_text = out_dir / "ref.txt"
    src_text.write_text("\n".join(src_lines) + "\n", encoding="utf-8")
    out_text.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    ref_text.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
    return src_text, out_text, ref_text


def evaluate(
    tests_dir: Path,
    model_dir: Path,
    langs=DEF_LANG_PAIRS,
    metrics=TASK_CONF["metrics"],
    test_names=None,
    batch_size: int = DEF_BATCH_SIZE,
    show_progress: bool = DEF_SHOW_PROGRESS,
    backup_dir: Path | None = None,
    run_num: int=1,
):

    run_cmd = get_run_cmd(model_dir)
    model_name = model_dir.name
    for pair in [normalize_lang_pair(lang) for lang in langs]:
        src, tgt = pair.split("-")
        lang_dir = tests_dir / pair
        pair_test_names = test_names
        if not pair_test_names:
            pair_test_names = [f.name.replace(f".{src}-{tgt}.jsonl", "") for f in lang_dir.glob(f"*.{src}-{tgt}.jsonl")]
            LOG.info(f"No test names specified. Using all available tests for {pair}: {pair_test_names}")
        for test_name in pair_test_names:
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.jsonl"
            if not src_file.exists():
                LOG.info(f"{test_name=} is unavailable for {pair}. {src_file} does not exist. Skipping.")
                continue
            ref = lang_dir / f"{test_name}.{src}-{tgt}.refs.jsonl"
            out = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}.{model_name}.out.batch{batch_size}.run{run_num}.jsonl"
            stats_file = out.with_suffix(out.suffix + ".stats.json")
            if not out.exists() or out.stat().st_size == 0:
                tmp_file = out.with_suffix(out.suffix + ".tmp")
                tmp_input_file = out.with_suffix(out.suffix + ".input.jsonl")
                tmp_file.unlink(missing_ok=True)
                write_participant_input(src_file, tmp_input_file)

                run_cmd_full = run_cmd + [
                    "--lang-pair",
                    pair,
                    "--batch-size",
                    str(batch_size),
                    "--input",
                    str(tmp_input_file),
                    "--output",
                    str(tmp_file),
                ]
                if show_progress:
                    run_cmd_full.append("--progress")
                LOG.info("Running command: %s", " ".join(run_cmd_full))
                try:
                    stats_start_time = time.time()
                    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    proc = sp.Popen(run_cmd_full)
                    proc.wait()
                    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    stats_end_time = time.time()
                    if proc.returncode != 0:
                        raise sp.CalledProcessError(proc.returncode, run_cmd_full)
                    if not tmp_file.exists() or tmp_file.stat().st_size == 0:
                        LOG.error("Submission did not write output file %s", tmp_file)
                        continue
                    try:
                        validate_jsonl_output(src_file, tmp_file)
                    except ValueError as exc:
                        LOG.error("Invalid JSONL output for %s: %s", tmp_file, exc)
                        tmp_file.unlink(missing_ok=True)
                        continue
                    stats = {
                        "run_num": run_num,
                        "model_name": model_name,
                        "batch_size": batch_size,
                        "out_file": str(out),
                        "job_name": os.getenv("JOB_NAME", os.getenv("SUB_ID", "N/A")),
                        "command": run_cmd_full,
                        "exit_code": proc.returncode,
                        "wall_time_sec": stats_end_time - stats_start_time,
                        "user_time_sec": r1.ru_utime - r0.ru_utime,
                        "sys_time_sec": r1.ru_stime - r0.ru_stime,
                        "max_rss_kb": r1.ru_maxrss,  # max resident set size (KB on Linux)
                        "inblock": r1.ru_inblock - r0.ru_inblock,
                        "oublock": r1.ru_oublock - r0.ru_oublock,
                        "voluntary_ctx_switches": r1.ru_nvcsw - r0.ru_nvcsw,
                        "involuntary_ctx_switches": r1.ru_nivcsw - r0.ru_nivcsw,
                        "start_timestamp": stats_start_time,
                        "end_timestamp": stats_end_time,
                    }
                    with open(stats_file, "a", encoding="utf-8") as sf:
                        sf.write(json.dumps(stats, ensure_ascii=False, indent=None) + "\n")

                    LOG.info(f"Wrote stats to {stats_file}")
                    tmp_file.rename(out)
                    LOG.info(f"Wrote translations to {out}")
                except sp.CalledProcessError as e:
                    LOG.error(f"Error running command: {e}")
                    continue
                finally:
                    tmp_input_file.unlink(missing_ok=True)
            for m in metrics:
                if not ref.exists() or ref.stat().st_size == 0:
                    LOG.info("Skipping %s for %s because reference file is missing", m, out)
                    continue
                score_file = out.with_suffix(out.suffix + f".{m}.score")
                if not score_file.exists() or score_file.stat().st_size == 0:
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            metric_files = write_metric_files(src_file, out, ref, Path(tmp_dir))
                            if metric_files is None:
                                LOG.info("Skipping %s for %s because references are missing", m, out)
                                continue
                            score = get_score(*metric_files, metric=m)
                        score_file.write_text(score)
                        LOG.info(f"{score_file.name} : {score}")
                    except sp.CalledProcessError as e:
                        LOG.error(f"Error scoring {out} with {m}: {e}")
                        continue
                else:
                    LOG.info(f"Skipping existing score file {score_file}")
            if backup_dir:
                backup_results(lang_dir, backup_dir / lang_dir.name)


def backup_results(from_dir:Path, to_dir:Path):
    """backup from from_dir to to_dir. Update new files. ignore existing and old files"""
    if not from_dir.exists():
        LOG.warning(f"Source directory {from_dir} does not exist")
        return
    LOG.info(f"Backing up results from {from_dir} --> {to_dir}")
    to_dir.mkdir(parents=True, exist_ok=True)
    copied, updated, skipped = 0, 0, 0
    for src in from_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(from_dir)
        dst = to_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dst.exists() or dst.stat().st_size == 0:
                shutil.copy2(src, dst)
                copied += 1
                LOG.info(f"Copied new file {rel}")
            else:
                if src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)
                    updated += 1
                    LOG.info(f"Updated file {rel}")
                else:
                    skipped += 1
        except OSError as e:
            LOG.error(f"Failed to copy {rel}: {e}")

    LOG.info(f"Backup summary: copied={copied} updated={updated} skipped={skipped}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WMT26 model-compression submissions", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR)
    parser.add_argument("-l", "--langs", nargs="+", help="Lang pairs to evaluate", default=DEF_LANG_PAIRS)
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=DEF_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to submission directory containing run.sh"
    )
    parser.add_argument("-t", "--test-names", nargs="+", help="Test names to evaluate; e.g. warmup. default: all tests available.", default=[])
    parser.add_argument(
        "-M", "--metrics", nargs="+", default=TASK_CONF["metrics"], help="Metrics to use for evaluation"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pbar", dest="show_progress", action="store_true", default=DEF_SHOW_PROGRESS,
        help="Enable progress bar during evaluation"
    )
    group.add_argument(
        "--no-pbar", dest="show_progress", action="store_false", default=DEF_SHOW_PROGRESS,
        help="Disable progress bar during evaluation"
    )

    job_name = os.environ.get("JOB_NAME") or os.environ.get("SUB_ID") or "local"
    def_backup_name = f"/mnt/tg/data/projects/wmt26/model-compression/evals/backup-v1/{job_name}"
    parser.add_argument(
        "-B", "--backup", type=Path, default=def_backup_name,
        help=f"Backup directory to save or update results. Use shared drive like blob container mount for archival purposes.")

    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of runs to perform")
    args = parser.parse_args()
    tests_dir = args.work / "tests"
    for i in range(1, args.runs + 1):
        LOG.info(f"Starting run {i}/{args.runs}")
        evaluate(tests_dir, args.model, langs=args.langs, batch_size=args.batch_size,
                test_names=args.test_names,
                metrics=args.metrics, show_progress=args.show_progress, backup_dir=args.backup,
                run_num=i)


if __name__ == "__main__":
    main()
