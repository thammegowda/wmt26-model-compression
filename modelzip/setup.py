#!/usr/bin/env python
"""Prepare local evaluation data for WMT26 model-compression runs."""

import argparse
import json
import logging as LOG
from pathlib import Path
from typing import Any

from modelzip.config import DEF_LANG_PAIRS, TASK_CONF, WORK_DIR, normalize_lang_pair
from modelzip.submission import INPUT_RECORD_KEYS, normalize_segment_text

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _reference_text(record: dict[str, Any]) -> str | None:
    refs = record.get("refs") or {}
    ref = refs.get("refA")
    if isinstance(ref, dict):
        ref = ref.get("ref")
    return ref if isinstance(ref, str) else None


def _normalize_rows(rows: list[Any], *, test_name: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return rows
    records = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, str):
            fields = [row]
        elif isinstance(row, (list, tuple)):
            fields = list(row)
        else:
            raise TypeError(f"Unexpected line format {type(row)} for {test_name}")
        if not fields or not fields[0]:
            continue
        record: dict[str, Any] = {
            "doc_id": f"{test_name}-{index}",
            "paragraph_id": 1,
            "src_text": fields[0],
        }
        if len(fields) > 1 and fields[1]:
            record["refs"] = {"refA": fields[1]}
        records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _participant_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: normalize_segment_text(record[key]) if key == "src_text" else record[key] for key in INPUT_RECORD_KEYS}


def setup_eval(work_dir: Path, langs=None):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = work_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    langs = [normalize_lang_pair(lang) for lang in (langs or DEF_LANG_PAIRS)]
    for lang_pair in langs:
        src, tgt = lang_pair.split("-")
        lang_dir = tests_dir / lang_pair
        lang_dir.mkdir(parents=True, exist_ok=True)
        for test_name, get_fn in TASK_CONF["langs"][lang_pair].items():
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.jsonl"
            ref_file = lang_dir / f"{test_name}.{src}-{tgt}.refs.jsonl"
            if src_file.exists() and src_file.stat().st_size > 0:
                LOG.info("Test files exist for %s:%s", lang_pair, test_name)
                continue
            LOG.info("Fetching %s via: %s", test_name, get_fn)
            rows = get_fn()
            assert isinstance(rows, list), f"Expected list of records, got {type(rows)}"
            records = _normalize_rows(rows, test_name=test_name)
            assert len(records) > 0, f"No records returned for {test_name} in {lang_pair}"

            participant_records = [_participant_record(record) for record in records]
            _write_jsonl(src_file, participant_records)
            LOG.info("Created source JSONL file %s", src_file)

            if any(_reference_text(record) is not None for record in records):
                ref_records = []
                for record in records:
                    ref_record = _participant_record(record)
                    ref_text = _reference_text(record)
                    ref_record["refs"] = {"refA": ref_text} if ref_text is not None else {}
                    ref_records.append(ref_record)
                _write_jsonl(ref_file, ref_records)
                LOG.info("Created reference JSONL file %s", ref_file)
            else:
                ref_file.unlink(missing_ok=True)
                LOG.info("Refs are missing for %s:%s", lang_pair, test_name)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WMT26 model-compression evaluation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR, help="Work directory")
    parser.add_argument("-l", "--langs", nargs="+", help="Language pairs to setup")
    parser.add_argument(
        "-t",
        "--task",
        choices=["eval"],
        default="eval",
        help="Compatibility option; root setup only prepares evaluation data",
    )
    args = parser.parse_args()
    setup_eval(work_dir=args.work, langs=args.langs)


if __name__ == "__main__":
    main()
