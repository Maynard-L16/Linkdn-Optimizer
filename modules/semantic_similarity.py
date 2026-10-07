"""Step 5 - Semantic similarity with Sentence-BERT embeddings + cosine similarity.

WHAT : Converts sentences into dense vectors (embeddings) whose direction
       encodes meaning, then measures the angle between profile and job.
WHY  : TF-IDF only sees exact words. "Built neural networks for image
       classification" and "experience with deep learning for computer vision"
       share no keywords but mean nearly the same thing. Embeddings capture
       this. We use a PRETRAINED model - nothing is trained here.
MODEL: sentence-transformers/all-MiniLM-L6-v2 (Reimers & Gurevych, 2019,
       "Sentence-BERT"). 6-layer MiniLM, 384-dim vectors, ~90 MB, runs on CPU.
       Max input 256 word-pieces per sentence, so we embed sentence by
       sentence instead of truncating a long profile.
INPUT: sentence lists from preprocessing (original wording, not lemmas -
       transformers understand natural text better than stop-word-free lemmas).
OUTPUT: SemanticResult - document-level cosine (used in the final score) and,
       for each job sentence, the most similar profile sentence.

Cosine similarity:  cos(a, b) = (a . b) / (||a|| * ||b||)   in [-1, 1]
Percentage shown  =  max(0, cos) * 100. Negative cosine (opposite meaning) is
rare for real text and is treated as 0 % similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_WORDS_PER_SENTENCE = 3  # headings like "Skills" carry no sentence meaning


@dataclass
class RequirementMatch:
    job_sentence: str
    best_profile_sentence: str
    similarity: float  # cosine, clipped to 0..1


@dataclass
class SemanticResult:
    document_similarity: float  # 0..1, used in the final score
    requirement_matches: list[RequirementMatch]
    model_name: str = MODEL_NAME


@lru_cache(maxsize=1)
def get_model():
    """Load the pretrained Sentence-BERT model once (downloaded on first use, then cached)."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the sentence-embedding model '{MODEL_NAME}'. The first run needs "
            "internet to download it (~90 MB); afterwards it is cached offline. Run:\n"
            "    python download_models.py\n"
            f"Original error: {exc}"
        ) from exc


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b) = a.b / (|a| |b|), written out explicitly for clarity."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def embed(sentences: list[str]) -> np.ndarray:
    """Return an (n_sentences x 384) matrix of sentence embeddings."""
    return np.asarray(get_model().encode(sentences, convert_to_numpy=True, show_progress_bar=False))


def _usable_sentences(sentences: list[str], fallback: str) -> list[str]:
    usable = [s for s in sentences if len(s.split()) >= MIN_WORDS_PER_SENTENCE]
    return usable or ([fallback] if fallback.strip() else [])


def analyze_semantics(job_sentences: list[str], profile_sentences: list[str], job_text: str, profile_text: str) -> SemanticResult:
    job_sents = _usable_sentences(job_sentences, job_text)
    profile_sents = _usable_sentences(profile_sentences, profile_text)
    if not job_sents or not profile_sents:
        return SemanticResult(document_similarity=0.0, requirement_matches=[])

    job_emb, profile_emb = embed(job_sents), embed(profile_sents)

    # Document vector = mean of its sentence vectors (mean pooling), so every
    # part of a long profile contributes instead of being truncated.
    doc_sim = cosine_similarity(job_emb.mean(axis=0), profile_emb.mean(axis=0))

    # Sentence-level view: for every job requirement, which profile sentence
    # addresses it best? (row-normalise, then one matrix product = all cosines)
    j_norm = job_emb / np.linalg.norm(job_emb, axis=1, keepdims=True)
    p_norm = profile_emb / np.linalg.norm(profile_emb, axis=1, keepdims=True)
    sim_matrix = j_norm @ p_norm.T
    matches = [
        RequirementMatch(
            job_sentence=job_sents[i],
            best_profile_sentence=profile_sents[int(np.argmax(sim_matrix[i]))],
            similarity=max(0.0, float(np.max(sim_matrix[i]))),
        )
        for i in range(len(job_sents))
    ]
    return SemanticResult(document_similarity=max(0.0, min(1.0, doc_sim)), requirement_matches=matches)
