#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
from typing import List

from modelzip.config import (
    DEF_BATCH_SIZE,
    DEF_MAX_NEW_TOKENS,
    LANGS_MAP,
    LOG,
    TRANSLATE_PROMPT,
    USE_CHAT_TEMPLATE,
    normalize_lang_pair,
)

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LLMWrapper:
    def __init__(
        self,
        model_dir: Path,
        use_chat_template=True,
        prompt_template=TRANSLATE_PROMPT,
        progress_bar=False,
        max_new_tokens=DEF_MAX_NEW_TOKENS,
    ):
        self.model_dir = Path(model_dir)
        self.use_chat_template = use_chat_template
        self.prompt_template = prompt_template
        self.progress_bar = progress_bar
        self.max_new_tokens = max_new_tokens
        self._config = None
        self._processor = None
        self._tokenizer = None
        self._model = None

    @property
    def config(self):
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_dir)
        return self._config

    @property
    def is_gemma3(self):
        return getattr(self.config, "model_type", "") == "gemma3"

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            loader_args = dict(device_map="auto", torch_dtype="auto")
            if self.is_gemma3:
                try:
                    from transformers import Gemma3ForConditionalGeneration
                except ImportError as exc:  # pragma: no cover - depends on the installed transformers version
                    raise RuntimeError(
                        "Gemma 3 requires transformers with Gemma3ForConditionalGeneration support"
                    ) from exc
                if Gemma3ForConditionalGeneration is None:
                    raise RuntimeError("Gemma 3 requires transformers with Gemma3ForConditionalGeneration support")
                model_cls = Gemma3ForConditionalGeneration
            else:
                model_cls = AutoModelForCausalLM
            self._model = model_cls.from_pretrained(self.model_dir, **loader_args)
            self._model.eval()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_dir)
            tokenizer = getattr(self._processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
        return self._processor

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def length_tokenizer(self):
        if self.is_gemma3:
            return self.processor.tokenizer
        return self.tokenizer

    @property
    def input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def _move_inputs(self, inputs):
        inputs = inputs.to(self.input_device) if hasattr(inputs, "to") else inputs
        if isinstance(inputs, dict):
            return {key: value.to(self.input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return inputs

    def _make_prompt(self, pair: str, text: str) -> str:
        src, tgt = pair.split("-")
        return self.prompt_template.format(src=LANGS_MAP[src], tgt=LANGS_MAP[tgt], text=text)

    def _generate_gemma3(self, prompts: list[str]) -> list[str]:
        import torch

        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in prompts]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
        )
        inputs = self._move_inputs(inputs)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        return [text.replace("\n", " ") for text in self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)]

    def _generate_causal_lm(self, prompts: list[str]) -> list[str]:
        import torch

        if self.use_chat_template:
            chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
            prompts = [
                self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in chats
            ]
        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=not self.use_chat_template,
        )
        inputs = self._move_inputs(inputs)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        return [text.replace("\n", " ") for text in self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)]

    def translate_lines(self, pair: str, lines: List[str], batch_size: int = DEF_BATCH_SIZE):
        from tqdm.auto import tqdm

        pair = normalize_lang_pair(pair)
        indexed = [(idx, line) for idx, line in enumerate(lines) if line.strip()]
        results = [""] * len(lines)
        if not indexed:
            return results

        # sort by token length
        def length(item):
            return len(self.length_tokenizer(item[1], add_special_tokens=False)["input_ids"])

        indexed.sort(key=length, reverse=True)

        batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]
        out_indexed = []
        pbar = tqdm(total=len(indexed), disable=not self.progress_bar)
        for batch in batches:
            ids, texts = zip(*batch)
            prompts = [self._make_prompt(pair, text) for text in texts]
            hyps = self._generate_gemma3(prompts) if self.is_gemma3 else self._generate_causal_lm(prompts)
            out_indexed.extend(zip(ids, hyps))
            pbar.update(len(batch))
        pbar.close()
        for idx, text in out_indexed:
            results[idx] = text
        return results


def main():
    args = parse_args()
    llm = LLMWrapper(
        args.model,
        use_chat_template=USE_CHAT_TEMPLATE,
        prompt_template=args.prompt,
        progress_bar=args.progress,
        max_new_tokens=args.max_new_tokens,
    )
    if args.input is sys.stdin:
        LOG.info("Reading from stdin")  # just in case if we forget to pass input via STDIN
    # buffering all inputs into one big maxibatch for sorting based on length,
    # assuming test sets are not too big
    lines = args.input.read().splitlines()
    assert len(lines) > 0, "Input file is empty. Please provide some input."
    outputs = llm.translate_lines(args.lang_pair, lines, batch_size=args.batch_size)
    assert len(outputs) == len(
        lines
    ), f"Output length {len(outputs)} does not match input length {len(lines)}"
    args.output.write("\n".join(outputs) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run translation using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pos_lang_pair", nargs="?", help="Compatibility positional language pair, e.g. ces-deu")
    parser.add_argument("pos_batch_size", nargs="?", type=int, help="Compatibility positional batch size")
    parser.add_argument("--lang-pair", help="Language pair to translate, e.g. ces-deu")
    parser.add_argument("--batch-size", type=int, help="Batch size for translation")

    # this script will/should be placed inside model directory for each model and called run.py,
    # so assume this file's parent dir as model dir
    my_name = Path(__file__).name
    my_dir = Path(__file__).parent
    default_model = Path(os.getenv("MODELZIP_MODEL_DIR", my_dir if my_name == "run.py" else "workdir/models/gemma-3-12b-it-base"))
    parser.add_argument(
        "-m",
        "--model",
        "--model-dir",
        dest="model",
        type=Path,
        default=default_model,
        help="Path to a Hugging Face Transformers-compatible model directory",
    )

    # optional args. Will not be set during evaluation, so make sure the defaults are correct
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
    parser.add_argument("--max-new-tokens", type=int, default=DEF_MAX_NEW_TOKENS)
    parser.add_argument(
        "-pt",
        "--prompt",
        type=str,
        default=TRANSLATE_PROMPT,
        help="Prompt template for translation",
    )
    args = parser.parse_args()
    lang_pair = args.lang_pair or args.pos_lang_pair
    if not lang_pair:
        parser.error("provide --lang-pair or the compatibility positional language pair")
    args.lang_pair = normalize_lang_pair(lang_pair)
    args.batch_size = args.batch_size or args.pos_batch_size or DEF_BATCH_SIZE
    return args


if __name__ == "__main__":
    main()
