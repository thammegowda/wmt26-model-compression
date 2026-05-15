# WMT26 Model Compression

This repository provides organizer baselines and a participant submission
template for the [WMT26 Model Compression shared task](https://www2.statmt.org/wmt26/model-compression.html).

The WMT26 task uses a simpler source-code submission format. Participants share a
complete runnable system, including installation instructions, inference code,
and the compressed model itself or a Hugging Face model repository link. Docker
images are no longer the primary submission artifact.

## Task Summary

- Constrained track: compress `google/gemma-3-12b-it`.
- Unconstrained track: compress any original model below 20B parameters.
- Language pairs: `ces-deu`, `eng-zho_Hans`, and `eng-ara_EG`.
- Evaluation hardware: Ubuntu 24.04 on x86_64 with one NVIDIA H100 GPU up to 80 GB VRAM.
- Evaluation criteria: translation quality, model size/on-disk and VRAM usage, and inference speed.

`google/gemma-3-12b-it` is gated on Hugging Face. Accept the Gemma terms and
provide access through `HF_TOKEN`, `huggingface-cli login`, or an authorized cache
before downloading the constrained baseline. Organizer nodes default to the
shared cache base `/mnt/tg/data/cache/tahoma/model-hub`, where Gemma 3 is
expected at `google/gemma-3-12b-it` under that directory.

## Submission Layout

Each submission should be a zip archive or Hugging Face repository with at least:

```text
setup.sh
run.sh
requirements.txt
pyproject.toml
modelzip/
```

The model may be included as files in the archive, placed under `workdir/models`,
or downloaded by `setup.sh` from a Hugging Face model repository.

### `setup.sh`

`setup.sh` creates a Python 3.12 environment with `uv`, installs this package,
and optionally downloads baseline models and development data. If `uv` is not
already available, the script attempts a user-local `python3 -m pip install uv`.
Model snapshots are resolved from `${MODELZIP_MODEL_CACHE_DIR}/MODEL_ID` and
downloaded there if missing. The default `MODELZIP_MODEL_CACHE_DIR` is
`/mnt/tg/data/cache/tahoma/model-hub`.

```bash
bash setup.sh
bash setup.sh --skip-model-download
bash setup.sh --model-id google/gemma-3-12b-it --lang-pair ces-deu
```

Useful options:

- `--model-id ID`: Hugging Face model ID to download.
- `--work-dir DIR`: directory for models and tests, default `workdir`.
- `--cache-dir DIR`: base cache directory for downloaded/source models.
- `--lang-pair PAIR`: language pair to prepare, repeatable.
- `--skip-model-download`: install code and data only.
- `--skip-eval-data`: install code and model only.

### `run.sh`

`run.sh` is the inference entry point used by the evaluator. The primary WMT26
interface is file-based:

```bash
bash run.sh \
  --lang-pair ces-deu \
  --batch-size 8 \
  --input input.txt \
  --output output.txt
```

The script must produce exactly one output line for each input line. Logs and
progress bars must go to stderr or separate files, not into the output file.

The compatibility positional interface still works for older WMT25-style model
directories:

```bash
bash model_dir/run.sh ces-deu 8 < input.txt > output.txt
```

## Development Data

Until WMT26 test data is released, this repo uses the WMT25 General MT blindset
as development data:

```text
https://data.statmt.org/wmt25/general-mt/wmt25.jsonl
```

That file is source-only. It contains `dataset_id`, `collection_id`, `doc_id`,
`domain`, `src_lang`, `tgt_lang`, `src_text`, optional `video`/`screenshot`, and
`prompt_instruction`, but no references. It is suitable for setup, inference,
line-count validation, and speed testing.

For local reference-based scoring, use the post-task WMT25 General MT GitHub
data instead:

```text
https://github.com/wmt-conference/wmt25-general-mt
```

The loader supports reference records with either `refs.refA.ref` or
`tgt_text.refA`.

## Baseline Workflow

Install and prepare the environment:

```bash
bash setup.sh --skip-model-download
```

Prepare development data:

```bash
python -m modelzip.setup --task eval --langs ces-deu eng-zho_Hans eng-ara_EG
```

Download the constrained baseline model after Hugging Face access is configured:

```bash
python -m modelzip.setup --task model --model-id google/gemma-3-12b-it
```

Run a baseline translation:

```bash
bash run.sh --lang-pair ces-deu --input input.txt --output output.txt --batch-size 1
```

Evaluate local outputs on available development sets:

```bash
python -m modelzip.evaluate -m workdir/models/gemma-3-12b-it-base -l ces-deu -b 1
```

Reference metrics are skipped automatically when the selected test set has no
reference file.

## Compression Demo

The sample compression script demonstrates BitsAndBytes quantization and writes
compressed model directories under `workdir/models`:

```bash
python -m modelzip.compress -m workdir/models/gemma-3-12b-it-base
```

Gemma 3 compatibility should be verified on the target H100 environment before
publishing baseline artifacts. `torchao` int4 is included as a candidate path for
future baselines, while BitsAndBytes variants are kept as a simple starting point.