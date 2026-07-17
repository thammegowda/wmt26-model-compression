#!/usr/bin/env python3
"""
Filter and split WMT26 gen-MT documents into paragraph-level JSONL records.

Reads wmt26 blind set (local path or URL), filters by source/target language,
and emits one JSONL line per paragraph. Paragraphs are delimited by blank lines
(\\n\\n) in src_text and in each reference.

Output fields per line: doc_id, paragraph_id, src_text, refs.

Usage:
  python split_paragraphs.py \\
    --src-lang cs \\
    --tgt-lang de_DE \\
    --jsonl https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/refs/heads/main/data/wmt25-genmt.jsonl \\
    -o wmt25.cs-de_DE.paragraphs.jsonl
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import TextIO, Tuple

DEFAULT_JSONL = "https://www2.statmt.org/wmt26/assets/wmt26_genmt_blindset.jsonl"
DOC_ID_MAPPING = "data/wmt26-meta/doc_id_mapping.jsonl"

EN_ARABIC_LANG_CODES = [('en', 'ar_AR'), ('en', 'arz'), ('eng_Latn', 'arz_Arab')]  # excluded ('en', 'aeb')
EN_CHINESE_LANG_CODES = [('en', 'zh_CN'), ('eng_Latn', 'zho_Hans')]  # excluded ('eng_Latn', 'zho_Hant_TW')
CS_GERMAN_LANG_CODES = [('ces_Latn', 'deu_Latn')]


def open_jsonl_stream(path: str) -> TextIO:
    if path.startswith("http://") or path.startswith("https://"):
        return urllib.request.urlopen(path)  # type: ignore[return-value]
    return open(path, encoding="utf-8")


def read_test_sets(path: str) -> dict[str, str]:
    test_sets = {}
    with open_jsonl_stream(path) as lines:
        for line in lines:
            doc = json.loads(line)
            test_sets[doc["doc_id"]] = doc["source_doc"]
    return test_sets


def read_doc_id_mapping(path: Path) -> dict[Tuple[str, str], list[str]]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            lang_pair = (doc["src_lang"], doc["tgt_lang"])
            if lang_pair not in records:
                records[lang_pair] = []
            records[lang_pair].append(doc["release_doc_id"])
    return records


def write_test_set(output_path: Path, docs: list[str], test_set: dict[str, str], src_lang: str, tgt_lang: str) -> None:
    with open(output_path / f"wmt26.{src_lang}.{tgt_lang}.jsonl", "w") as fh:
        for doc in docs:
            fh.write(json.dumps({"doc_id": doc, "src_text": test_set[doc]}, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default=DEFAULT_JSONL, help="Input JSONL path or URL")
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSONL folder",
    )
    args = ap.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=False)

    # root directory of the repository
    repo_base = Path(__file__).resolve().parent.parent

    # Relative path to the JSONL file
    doc_id_mapping = read_doc_id_mapping(repo_base / DOC_ID_MAPPING)

    test_sets = read_test_sets(args.jsonl)

    for lp_codes in [EN_ARABIC_LANG_CODES, EN_CHINESE_LANG_CODES, CS_GERMAN_LANG_CODES]:
        try:
            docs = []
            src_lang = lp_codes[0][0]
            tgt_lang = lp_codes[0][1]
            for lp_code in lp_codes:
                docs.extend(doc_id_mapping[lp_code])
            write_test_set(output_path, docs, test_sets, src_lang, tgt_lang)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Wrote {len(docs)} documents ({src_lang} -> {tgt_lang})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
