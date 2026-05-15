import logging as LOG
import os
from pathlib import Path
from .data import CmdGetter, Wmt25BlindData, Wmt25ReferenceData, WmtJsonlData


LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_MODEL_CACHE_DIR = "/mnt/tg/data/cache/tahoma/model-hub"
MODEL_CACHE_DIR = Path(os.getenv("MODELZIP_MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)).expanduser()
HF_CACHE = Path(os.getenv("HF_HOME", default=Path.home() / ".cache" / "huggingface")) / "hub"
WORK_DIR = "./workdir"

# reusing hf cache for data; not perfect but for simpler config sake
WmtJsonlData.CACHE_DIR = HF_CACHE / "wmt26-modelzip"

WMT25_BLIND_URL = os.getenv("MODELZIP_WMT25_BLIND_URL", Wmt25BlindData.URL)
WMT25_REF_URL = os.getenv("MODELZIP_WMT25_REF_URL", Wmt25ReferenceData.URL)
WMT26_DATA_URL = os.getenv("MODELZIP_WMT26_DATA_URL", "")

CONSTRAINED_MODEL = os.getenv("MODELZIP_MODEL_ID", "google/gemma-3-12b-it")


# Task configuration
TASK_CONF = {
    "langs": {
        "ces-deu": {
            "warmup": CmdGetter("printf 'ahoj světe\tHallo Welt\n'"),
            "wmt25-blind": Wmt25BlindData("cs-de_DE", url=WMT25_BLIND_URL),
            "wmt25-ref": Wmt25ReferenceData("cs-de_DE", url=WMT25_REF_URL),
        },
        "eng-zho_Hans": {
            "warmup": CmdGetter("printf 'hello world\t你好，世界\n'"),
            "wmt25-blind": Wmt25BlindData("en-zh_CN", url=WMT25_BLIND_URL),
            "wmt25-ref": Wmt25ReferenceData("en-zh_CN", url=WMT25_REF_URL),
        },
        "eng-ara_EG": {
            "warmup": CmdGetter("printf 'hello world\tمرحبا بالعالم\n'"),
            "wmt25-blind": Wmt25BlindData("en-ar_EG", url=WMT25_BLIND_URL),
            "wmt25-ref": Wmt25ReferenceData("en-ar_EG", url=WMT25_REF_URL),
        },
    },
    "models": [CONSTRAINED_MODEL],
    "metrics": ["chrf", "wmt22-comet-da"],  # "wmt22-cometkiwi-da" is a gated model
}

# Default language pairs
DEF_LANG_PAIRS = list(TASK_CONF["langs"].keys())
# Default batch size for translation
DEF_BATCH_SIZE = 1

# Mapping of language codes to full names
LANGS_MAP = dict(
    ces="Czech",
    deu="German",
    zho="Chinese",
    zho_Hans="Simplified Chinese",
    zh_CN="Simplified Chinese",
    eng="English",
    ara="Arabic",
    ara_EG="Egyptian Arabic",
    ar_EG="Egyptian Arabic",
)

# two-letter language codes just in case we need them
_aliases = """
ces cs
deu de
zho zh
eng en
ara ar
""".strip()
for line in _aliases.splitlines():
    code3, code2 = line.split()
    LANGS_MAP[code2] = LANGS_MAP[code3]


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


def normalize_lang_pair(pair: str) -> str:
    normalized = LANG_PAIR_ALIASES.get(pair, pair)
    if normalized not in TASK_CONF["langs"]:
        known = ", ".join(sorted(TASK_CONF["langs"]))
        aliases = ", ".join(sorted(LANG_PAIR_ALIASES))
        raise ValueError(f"Unsupported language pair {pair!r}. Known pairs: {known}. Aliases: {aliases}")
    return normalized


# Default translation prompt template
TRANSLATE_PROMPT = (
    "Translate the following text from {src} to {tgt}. "
    "Return only the translation, with no explanation, labels, or quotes.\n\n"
    "{text}\n"
)
USE_CHAT_TEMPLATE = True
DEF_MAX_NEW_TOKENS = int(os.getenv("MODELZIP_MAX_NEW_TOKENS", "1024"))
