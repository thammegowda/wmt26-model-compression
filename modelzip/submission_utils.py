from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, TextIO

DEF_BATCH_SIZE = 1
DEF_MAX_NEW_TOKENS = int(os.getenv("MODEL_MAX_NEW_TOKENS", "1024"))

DEF_LANG_PAIRS = ["ces-deu", "eng-zho_Hans", "eng-ara_EG"]

LANGS_MAP = {
    "ces": "Czech",
    "deu": "German",
    "zho_Hans": "Simplified Chinese",
    "zh_CN": "Simplified Chinese",
    "eng": "English",
    "ara_EG": "Egyptian Arabic",
    "ar_EG": "Egyptian Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "zh": "Simplified Chinese",
    "ar": "Egyptian Arabic",
}

LANG_PAIR_ALIASES = {
    "cs-de": "ces-deu",
    "cs-de_DE": "ces-deu",
    "ces-deu": "ces-deu",
    "en-zh": "eng-zho_Hans",
    "en-zh_CN": "eng-zho_Hans",
    "eng-zho": "eng-zho_Hans",
    "eng-zho_Hans": "eng-zho_Hans",
    "en-ar": "eng-ara_EG",
    "en-ar_EG": "eng-ara_EG",
    "eng-ara": "eng-ara_EG",
    "eng-ara_EG": "eng-ara_EG",
}

TRANSLATE_PROMPT = (
    "Translate the following text from {src} to {tgt}. "
    "Return only the translation, with no explanation, labels, or quotes.\n\n"
    "{text}\n"
)


def normalize_lang_pair(pair: str) -> str:
    normalized = LANG_PAIR_ALIASES.get(pair, pair)
    if normalized not in DEF_LANG_PAIRS:
        known = ", ".join(sorted(DEF_LANG_PAIRS))
        aliases = ", ".join(sorted(LANG_PAIR_ALIASES))
        raise ValueError(f"Unsupported language pair {pair!r}. Known pairs: {known}. Aliases: {aliases}")
    return normalized


def language_names(pair: str) -> tuple[str, str]:
    src, tgt = normalize_lang_pair(pair).split("-")
    return LANGS_MAP[src], LANGS_MAP[tgt]


def make_translation_prompt(pair: str, text: str, template: str = TRANSLATE_PROMPT) -> str:
    src, tgt = language_names(pair)
    return template.format(src=src, tgt=tgt, text=text)


def read_source_lines(input_file: TextIO) -> list[str]:
    lines = input_file.read().splitlines()
    if not lines:
        raise ValueError("Input file is empty. Please provide some input.")
    return lines


def write_output_lines(output_file: TextIO, lines: Iterable[str]) -> None:
    output_file.write("\n".join(line.replace("\n", " ") for line in lines) + "\n")


def validate_line_count(inputs: list[str], outputs: list[str]) -> None:
    if len(outputs) != len(inputs):
        raise ValueError(f"Output length {len(outputs)} does not match input length {len(inputs)}")


def parse_inference_args(
    *,
    default_model: Path,
    description: str = "Run translation using a submission model",
    default_prompt: str = TRANSLATE_PROMPT,
    default_batch_size: int = DEF_BATCH_SIZE,
    default_max_new_tokens: int = DEF_MAX_NEW_TOKENS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pos_lang_pair", nargs="?", help="Compatibility positional language pair, e.g. ces-deu")
    parser.add_argument("pos_batch_size", nargs="?", type=int, help="Compatibility positional batch size")
    parser.add_argument("--lang-pair", help="Language pair to translate, e.g. ces-deu")
    parser.add_argument("--batch-size", type=int, help="Batch size for translation")
    parser.add_argument(
        "-m",
        "--model",
        "--model-dir",
        dest="model",
        type=Path,
        default=default_model,
        help="Path to a Hugging Face Transformers-compatible model directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r", encoding="utf-8", errors="replace"),
        default=sys.stdin,
        help="Input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="utf-8", errors="replace"),
        default=sys.stdout,
        help="Output file",
    )
    parser.add_argument("-pb", "--progress", action="store_true", help="Show progress bar")
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("-pt", "--prompt", type=str, default=default_prompt, help="Prompt template for translation")
    args = parser.parse_args()

    lang_pair = args.lang_pair or args.pos_lang_pair
    if not lang_pair:
        parser.error("provide --lang-pair or the compatibility positional language pair")
    try:
        args.lang_pair = normalize_lang_pair(lang_pair)
    except ValueError as exc:
        parser.error(str(exc))
    args.batch_size = args.batch_size or args.pos_batch_size or default_batch_size
    return args