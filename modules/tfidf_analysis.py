"""Step 3 - TF-IDF keyword analysis.

WHAT : Weighs every word / two-word phrase by TF-IDF, finds the most important
       job-description keywords, checks which of them the profile contains,
       and computes the classic TF-IDF cosine similarity of the two texts.
WHY  : Recruiters and ATS (applicant tracking) systems look for the job's key
       terms. TF-IDF is the textbook way to find which terms characterise a
       document (Salton & Buckley, 1988; Manning, Raghavan & Schuetze,
       "Introduction to Information Retrieval", 2008, ch. 6).
INPUT: two PreprocessResult objects (lemmatised, stop-words removed).
OUTPUT: TfidfResult - ranked job keywords with weights, matched/missing
       keywords, keyword coverage (used in the final score) and TF-IDF cosine.

Formula (identical to scikit-learn's TfidfVectorizer with smooth_idf=True,
sublinear_tf=True, norm="l2" - verified in tests/test_tfidf_analysis.py):
    tf(t, d) = 1 + ln(count of t in d)        sublinear: 5 mentions != 5x as important
    idf(t)   = ln((1 + N) / (1 + df(t))) + 1  N = documents in the reference corpus,
                                              df(t) = how many of them contain t
    w(t, d)  = tf * idf, then the vector is divided by its length (L2 norm)

Where does IDF come from?
    IDF needs MANY documents, so it is taken from a reference corpus of real
    public job postings (data/background_idf.json, built by
    scripts/build_background_idf.py). Words used in almost every posting
    ("team", "experience", "work") get a LOW idf; specific terms ("pytorch",
    "data analysis") get a HIGH idf. Terms never seen in the corpus get the
    maximum idf. If the file is missing, idf = 1 for every term (pure TF).

The TF-IDF arithmetic is written out by hand below (sparse dict vectors), so
every number shown in the UI can be traced back to this formula.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from modules.nlp_resources import DATA_DIR
from modules.preprocessing import PreprocessResult

BACKGROUND_IDF_PATH = DATA_DIR / "background_idf.json"
TOP_N_KEYWORDS = 15
# A two-word phrase may only be offered as a keyword if it is an established
# phrase: a known skill phrase, or used in at least this many reference
# postings (a phrase repeated inside ONE posting, like "New York hub", is not
# enough). Otherwise rare accidental word pairs
# ("diverse enterprise") get the maximum IDF and crowd out real keywords -
# a known weakness of IDF on sparse n-grams.
MIN_PHRASE_DF = 5
SEGMENT_SEPARATOR = " | "

Vector = dict[str, float]


@dataclass(frozen=True)
class Background:
    n_documents: int
    df: dict[str, int]
    description: str


@dataclass
class Keyword:
    term: str
    weight: float  # TF-IDF weight in the job description
    in_profile: bool
    match_type: str = ""  # "exact", "synonym" (via skill aliases) or "" when missing
    idf: float = 0.0
    count: int = 0  # occurrences in the job description


@dataclass
class TfidfResult:
    job_keywords: list[Keyword]
    matched_keywords: list[str]
    missing_keywords: list[str]
    keyword_coverage: float | None  # weighted share of top job keywords found in profile (0..1)
    tfidf_cosine: float  # cosine similarity of full TF-IDF vectors (0..1)
    profile_top_terms: list[tuple[str, float]]
    comparison: list[dict[str, float | str]]  # term, job weight, profile weight
    idf_source: str


# --------------------------------------------------------------------------- terms
def to_document(segments: list[list[str]]) -> str:
    """Serialise segments as 'machine learning | python'."""
    return SEGMENT_SEPARATOR.join(" ".join(seg) for seg in segments if seg)


def segment_ngrams(document: str) -> Iterator[str]:
    """Unigrams + bigrams; a bigram never crosses a segment boundary."""
    for segment in document.split("|"):
        words = segment.split()
        yield from words
        for a, b in zip(words, words[1:]):
            yield f"{a} {b}"


# --------------------------------------------------------------------------- TF-IDF maths
@lru_cache(maxsize=4)
def load_background(path: Path = BACKGROUND_IDF_PATH) -> Background:
    if not Path(path).exists():
        return Background(0, {}, "no reference corpus found - idf = 1 (pure term frequency)")
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    meta = raw.get("meta", {})
    description = (
        f"{raw['n_documents']:,} public job postings from {len(meta.get('boards', []))} companies "
        f"({meta.get('source', 'unknown source')}, built {meta.get('built_on', '?')})"
    )
    return Background(raw["n_documents"], raw["df"], description)


def idf(term: str, bg: Background) -> float:
    if bg.n_documents == 0:
        return 1.0
    return math.log((1 + bg.n_documents) / (1 + bg.df.get(term, 0))) + 1


def tfidf_vector(document: str, bg: Background) -> tuple[Vector, Counter[str]]:
    """Return the L2-normalised TF-IDF vector of a document, and its raw term counts."""
    counts = Counter(segment_ngrams(document))
    weights = {t: (1 + math.log(c)) * idf(t, bg) for t, c in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values()))
    return ({t: w / norm for t, w in weights.items()} if norm else {}), counts


def cosine(a: Vector, b: Vector) -> float:
    """Both vectors are unit length, so cosine = dot product over shared terms."""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


# --------------------------------------------------------------------------- keyword ranking
def _term_phrases(counts: Counter[str], known_phrases: set[str], bg: Background) -> set[str]:
    """Bigrams that behave like a single term: a known multi-word skill
    ("data analysis") or an established phrase in the reference corpus.
    Without a corpus (idf = 1) a phrase repeated in the text is accepted."""
    return {
        t for t, c in counts.items()
        if " " in t and (t in known_phrases or bg.df.get(t, 0) >= MIN_PHRASE_DF or (bg.n_documents == 0 and c >= 2))
    }


def top_terms(
    vector: Vector,
    n: int,
    exclude: set[str] | frozenset[str] = frozenset(),
    term_phrases: set[str] | frozenset[str] = frozenset(),
) -> list[tuple[str, float]]:
    """Rank terms by TF-IDF weight.

    * Only established phrases (see MIN_PHRASE_DF) compete as bigrams.
    * Unigrams that are part of a selected TERM phrase are dropped ("machine"
      is redundant next to "machine learning"). An incidental bigram such as
      "docker basics" does not hide the important word "docker".
    * Ties at the cut-off are all kept (a short job description has many
      equal weights; cutting them alphabetically would be arbitrary). A hard
      cap of 2n keeps the list readable.
    """
    ranked = sorted(
        ((t, w) for t, w in vector.items() if t not in exclude and (" " not in t or t in term_phrases)),
        key=lambda tw: (-round(tw[1], 9), tw[0]),
    )
    phrase_words = {word for t, _ in ranked[: 3 * n] if t in term_phrases for word in t.split()}
    ranked = [(t, w) for t, w in ranked if " " in t or t not in phrase_words]

    picked: list[tuple[str, float]] = []
    for term, weight in ranked:
        tie = picked and abs(weight - picked[-1][1]) < 1e-9
        if len(picked) >= 2 * n or (len(picked) >= n and not tie):
            break
        picked.append((term, weight))
    return picked


def analyze_keywords(
    job: PreprocessResult,
    profile: PreprocessResult,
    top_n: int = TOP_N_KEYWORDS,
    synonym_groups: list[set[str]] | None = None,
    known_phrases: set[str] | None = None,
    background: Background | None = None,
    exclude_terms: set[str] | None = None,
) -> TfidfResult:
    """synonym_groups: alias sets of skills found in the PROFILE (e.g. {"ai",
    "artificial intelligence"}). A job keyword counts as present when the
    profile uses a synonym of it - without this, pure lexical matching misses
    "AI" vs "Artificial Intelligence".
    known_phrases: multi-word skill names treated as single terms when ranking.
    exclude_terms: never offered as keywords (e.g. the employer's own name).
    """
    bg = background or load_background()
    synonyms = {alias for group in (synonym_groups or []) for alias in group}
    phrases = known_phrases or set()

    job_vec, job_counts = tfidf_vector(to_document(job.segments), bg)
    profile_vec, profile_counts = tfidf_vector(to_document(profile.segments), bg)

    job_keywords = []
    job_exclude = job.non_phrase_lemmas | (exclude_terms or set())
    for term, weight in top_terms(job_vec, top_n, job_exclude, _term_phrases(job_counts, phrases, bg)):
        match_type = "exact" if term in profile_counts else "synonym" if term in synonyms else ""
        job_keywords.append(Keyword(term, weight, bool(match_type), match_type, idf(term, bg), job_counts[term]))

    total = sum(k.weight for k in job_keywords)
    coverage = sum(k.weight for k in job_keywords if k.in_profile) / total if total > 0 else None

    profile_top = top_terms(profile_vec, top_n, profile.non_phrase_lemmas, _term_phrases(profile_counts, phrases, bg))
    terms = list(dict.fromkeys([k.term for k in job_keywords] + [t for t, _ in profile_top]))
    comparison = [
        {"term": t, "job_weight": round(job_vec.get(t, 0.0), 4), "profile_weight": round(profile_vec.get(t, 0.0), 4)}
        for t in terms
    ]

    return TfidfResult(
        job_keywords=job_keywords,
        matched_keywords=[k.term for k in job_keywords if k.in_profile],
        missing_keywords=[k.term for k in job_keywords if not k.in_profile],
        keyword_coverage=coverage,
        tfidf_cosine=max(0.0, min(1.0, cosine(job_vec, profile_vec))),
        profile_top_terms=profile_top,
        comparison=comparison,
        idf_source=bg.description,
    )
