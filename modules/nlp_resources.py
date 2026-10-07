"""Shared NLP resources: file paths, the spaCy pipeline and the stop-word list.

Models are loaded ONCE and cached (functools.lru_cache) because loading spaCy
or a transformer model takes seconds, while analysing a text takes milliseconds.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"

SKILLS_PATH = DATA_DIR / "skills.json"
QUALIFICATIONS_PATH = DATA_DIR / "qualifications.json"
CUSTOM_STOPWORDS_PATH = DATA_DIR / "custom_stopwords.txt"
BOILERPLATE_CUES_PATH = DATA_DIR / "jd_boilerplate_cues.txt"

SPACY_MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load spaCy's small English pipeline and add a rule-based QUALIFICATION recogniser.

    en_core_web_sm gives us: tokenizer, POS tagger, dependency parser (used for
    sentence boundaries), lemmatizer and a statistical NER trained on OntoNotes 5.
    OntoNotes has no label for degrees/certifications, so we add an EntityRuler
    (pattern matching) *before* the statistical NER; the NER then respects the
    spans the ruler already labelled.
    """
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL_NAME}' is not installed. Run:\n"
            f"    python -m spacy download {SPACY_MODEL_NAME}\n"
            "or:  python download_models.py"
        ) from exc

    # The default tokenizer splits "C#" into "C" + "#"; keep such names whole.
    for name in ("C#", "c#", "F#", "f#"):
        nlp.tokenizer.add_special_case(name, [{"ORTH": name}])

    with QUALIFICATIONS_PATH.open(encoding="utf-8") as f:
        qual = json.load(f)
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True},
    )
    ruler.add_patterns(
        [{"label": qual["label"], "pattern": p, "id": "rule_based"} for p in qual["phrases"]]
    )
    return nlp


@lru_cache(maxsize=1)
def get_stopwords() -> tuple[frozenset[str], str]:
    """Return (stop-word set, description of its source).

    Base list: NLTK's English stop-word list (~200 words). If the NLTK corpus is not
    downloaded and cannot be downloaded (offline), fall back to spaCy's built-in
    list so the app still works. Domain stop-words from data/custom_stopwords.txt
    are always added.
    """
    source = "NLTK English stop-words"
    try:
        from nltk.corpus import stopwords

        base = set(stopwords.words("english"))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords

            base = set(stopwords.words("english"))
        except Exception:  # offline or download blocked
            from spacy.lang.en.stop_words import STOP_WORDS

            base = set(STOP_WORDS)
            source = "spaCy English stop-words (NLTK corpus unavailable)"

    custom = set(read_word_list(CUSTOM_STOPWORDS_PATH))
    return frozenset(base | custom), f"{source} + {len(custom)} domain stop-words"


def read_word_list(path: Path) -> list[str]:
    """One entry per line, lower-cased; blank lines and # comments ignored."""
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
