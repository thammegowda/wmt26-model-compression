#!/usr/bin/env python
#
# 2025-05-09: Initial version by TG Gowda
# 2025-08-17:  add support for source only wmt25 testsets
#
"""
Setup script for WMT26 model-compression baselines.

This script downloads baseline models and development sets. The WMT25 General MT
blindset is used as development data until WMT26 test data is released.
"""

import argparse
import logging as LOG
from pathlib import Path

from modelzip.config import DEF_LANG_PAIRS, MODEL_CACHE_DIR, TASK_CONF, WORK_DIR, normalize_lang_pair

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.{src}"
            ref_file = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}"
            meta_file = lang_dir / f"{test_name}.{src}-{tgt}.meta"
            if src_file.exists() and src_file.stat().st_size > 0 and (ref_file.exists() or meta_file.exists()):
                LOG.info(f"Test files exist for {lang_pair}:{test_name}")
                continue
            LOG.info(f"Fetching {test_name} via: {get_fn}")
            lines = get_fn()
            assert isinstance(lines, list), f"Expected list of lines, got {type(lines)}"
            assert len(lines) > 0, f"No lines returned for {test_name} in {lang_pair}"
            if isinstance(lines[0], str):
                src_file.write_text("\n".join(lines))
                LOG.info(f"Created test files {src_file}; NOTE: refs are missing")
            elif isinstance(lines[0], (list, tuple)):
                n_fields = len(lines[0])
                srcs = [x[0] for x in lines]
                src_file.write_text("\n".join(srcs))
                LOG.info(f"Created source file {src_file}")
                if n_fields > 1:
                    refs = [x[1] for x in lines]
                    # tgt can be None
                    if all(ref is None for ref in refs):  # all tgt segs are None
                        LOG.info("Refs are missing")
                    else:  # no tgt seg is None
                        assert all(ref is not None for ref in refs), "Some references are None"
                        ref_file.write_text("\n".join(refs))
                        LOG.info(f"Created ref_file {ref_file}")
                if n_fields > 2:  # src, tgt, meta
                    meta = [x[2] for x in lines]
                    meta_file.write_text("\n".join(meta))
                    LOG.info(f"Created meta file {meta_file}")
            else:
                LOG.info(
                    f"Unexpected line format {type(lines[0])} for {test_name} in {lang_pair}. "
                    "str, List[str], or tuple[str] expected"
                )


def _model_basename(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def _resolve_model_source(model_id: str, cache_dir: Path) -> Path:
    explicit_path = Path(model_id).expanduser()
    if explicit_path.exists():
        return explicit_path
    if explicit_path.is_absolute():
        raise FileNotFoundError(f"Model path does not exist: {explicit_path}")

    cached_model_dir = Path(cache_dir).expanduser() / model_id
    if (cached_model_dir / "config.json").exists():
        LOG.info("Using cached model %s from %s", model_id, cached_model_dir)
        return cached_model_dir

    cached_model_dir.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s to %s", model_id, cached_model_dir)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, local_dir=cached_model_dir)
    (cached_model_dir / "._DOWNLOAD_OK").touch()
    return cached_model_dir


def _load_processor_or_tokenizer(model_source: Path):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    if getattr(config, "model_type", "") == "gemma3":
        return AutoProcessor.from_pretrained(model_source, local_files_only=True), config
    return AutoTokenizer.from_pretrained(model_source, local_files_only=True, use_fast=True), config


def _load_model(model_source: Path, config):
    from transformers import AutoModelForCausalLM

    loader_args = dict(local_files_only=True, device_map="auto", torch_dtype="auto")
    if getattr(config, "model_type", "") == "gemma3":
        try:
            from transformers import Gemma3ForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on installed transformers version
            raise RuntimeError("Gemma 3 requires transformers with Gemma3ForConditionalGeneration support") from exc
        model_cls = Gemma3ForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM
    LOG.info("Loading model from %s; args: %s", model_source, loader_args)
    return model_cls.from_pretrained(model_source, **loader_args)


def setup_model(work_dir: Path, cache_dir: Path, model_ids=TASK_CONF["models"]):
    # downloads
    work_dir = Path(work_dir)
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        simple_name = _model_basename(model_id)
        model_dir = models_dir / (simple_name + "-base")
        model_dir.mkdir(parents=True, exist_ok=True)
        flag_file = model_dir / "._OK"
        if flag_file.exists():
            LOG.info(f"Model {model_id} already exists; rm {flag_file} to force download")
            continue

        model_source = _resolve_model_source(model_id, cache_dir=cache_dir)
        processor_or_tokenizer, config = _load_processor_or_tokenizer(model_source)
        model = _load_model(model_source, config=config)
        LOG.info(f"{model_id} loaded successfully. Storing at {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        processor_or_tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        # copy baseline.py as run.py inside the model_dir
        run_script = model_dir / "run.sh"
        baseline_script = Path(__file__).parent / "run.sh"
        assert baseline_script.exists(), f"Baseline script {baseline_script} does not exist"
        run_script.write_text(baseline_script.read_text())

        flag_file.touch()
        LOG.info(f"Model {model_id} saved successfully at {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Setup WMT26 model-compression baselines",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR, help="Work directory")
    parser.add_argument("-l", "--langs", nargs="+", help="Language pairs to setup")
    parser.add_argument("--model-id", nargs="+", default=TASK_CONF["models"], help="Hugging Face model IDs to download")
    parser.add_argument(
        "-t",
        "--task",
        choices=["eval", "model", "all"],
        default="all",
        help="Task to perform",
    )
    parser.add_argument(
        "-c",
        "--cache",
        "--cache-dir",
        "--model-cache-dir",
        dest="cache",
        type=Path,
        default=MODEL_CACHE_DIR,
        help="Base cache directory for downloaded/source models",
    )
    args = parser.parse_args()

    # dispatch based on task
    if args.task in ("model", "all"):
        setup_model(work_dir=args.work, cache_dir=args.cache, model_ids=args.model_id)
    if args.task in ("eval", "all"):
        setup_eval(work_dir=args.work, langs=args.langs)


if __name__ == "__main__":
    main()
