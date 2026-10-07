"""One-time setup: download every pretrained resource the app needs.

    python download_models.py

Downloads (only if missing):
  1. spaCy English pipeline  en_core_web_sm   (~13 MB)
  2. NLTK stop-word list                       (<1 MB)
  3. Sentence-BERT model all-MiniLM-L6-v2      (~90 MB, cached by Hugging Face)

After this, the app runs fully offline.
"""

from __future__ import annotations

import subprocess
import sys


def ensure_spacy_model() -> None:
    try:
        import en_core_web_sm  # noqa: F401

        print("[1/3] spaCy model en_core_web_sm already installed.")
    except ImportError:
        print("[1/3] Downloading spaCy model en_core_web_sm ...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)


def ensure_nltk_stopwords() -> None:
    import nltk

    try:
        from nltk.corpus import stopwords

        stopwords.words("english")
        print("[2/3] NLTK stop-words already available.")
    except LookupError:
        print("[2/3] Downloading NLTK stop-words ...")
        nltk.download("stopwords")


def ensure_sentence_model() -> None:
    from modules.semantic_similarity import MODEL_NAME, get_model

    print(f"[3/3] Loading / downloading {MODEL_NAME} ...")
    model = get_model()
    print(f"      ready - embedding size {model.encode(['test']).shape[1]}.")


if __name__ == "__main__":
    ensure_spacy_model()
    ensure_nltk_stopwords()
    ensure_sentence_model()
    print("All models ready. Start the app with:  streamlit run app.py")
