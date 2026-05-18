#!/usr/bin/env python
import logging as LOG
import os
from pathlib import Path
from typing import Any, List

from modelzip.submission_utils import (
    DEF_BATCH_SIZE,
    DEF_MAX_NEW_TOKENS,
    TRANSLATE_PROMPT,
    make_translation_prompt,
    normalize_lang_pair,
    parse_inference_args,
    read_source_lines,
    validate_line_count,
    write_output_lines,
)

USE_CHAT_TEMPLATE = True

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

            self._config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        return self._config

    @property
    def is_gemma3(self):
        return getattr(self.config, "model_type", "") == "gemma3"

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            loader_args: dict[str, Any] = dict(device_map="auto", torch_dtype="auto", local_files_only=True)
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

            self._processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
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

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True, local_files_only=True)
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
        return make_translation_prompt(pair, text, self.prompt_template)

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
    lines = read_source_lines(args.input)
    outputs = llm.translate_lines(args.lang_pair, lines, batch_size=args.batch_size)
    validate_line_count(lines, outputs)
    write_output_lines(args.output, outputs)


def parse_args():
    my_dir = Path(__file__).parent
    default_model = Path(os.getenv("MODEL_DIR", os.getenv("MODELZIP_MODEL_DIR", my_dir / "workdir" / "model")))
    return parse_inference_args(
        default_model=default_model,
        description="Run translation using the BNB q8 Gemma baseline",
        default_prompt=TRANSLATE_PROMPT,
        default_max_new_tokens=DEF_MAX_NEW_TOKENS,
    )


if __name__ == "__main__":
    main()
