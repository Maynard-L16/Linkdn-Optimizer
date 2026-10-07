import math

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.preprocessing import preprocess
from modules.tfidf_analysis import (
    Background,
    analyze_keywords,
    idf,
    load_background,
    segment_ngrams,
    tfidf_vector,
    to_document,
)

NO_IDF = Background(0, {}, "test: idf = 1")


def test_segment_ngrams_never_cross_boundaries():
    doc = to_document([["python"], ["machine", "learning"]])
    assert doc == "python | machine learning"
    assert list(segment_ngrams(doc)) == ["python", "machine", "learning", "machine learning"]


def test_idf_formula():
    bg = Background(9, {"team": 9, "pytorch": 1}, "test")
    assert idf("team", bg) == pytest.approx(math.log(10 / 10) + 1)  # in every doc -> minimum idf 1.0
    assert idf("pytorch", bg) == pytest.approx(math.log(10 / 2) + 1)
    assert idf("unseen", bg) == pytest.approx(math.log(10 / 1) + 1)  # never seen -> maximum idf


def test_manual_tfidf_equals_scikit_learn():
    """Our hand-written TF-IDF must give exactly scikit-learn's numbers."""
    corpus = ["python | machine learning", "python | sql | team", "team | communication", "machine learning | team"]
    analyzer = lambda d: list(segment_ngrams(d))  # noqa: E731
    sk = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, smooth_idf=True, norm="l2").fit(corpus)
    df = {t: sum(t in set(analyzer(d)) for d in corpus) for t in sk.get_feature_names_out()}
    bg = Background(len(corpus), df, "test")

    doc = "python | python | machine learning | sql"
    expected = dict(zip(sk.get_feature_names_out(), sk.transform([doc]).toarray()[0]))
    ours, _ = tfidf_vector(doc, bg)
    for term, weight in expected.items():
        assert ours.get(term, 0.0) == pytest.approx(weight)


def test_identical_texts_have_full_similarity_and_coverage():
    text = "Python developer with machine learning experience. Builds NLP models."
    r = analyze_keywords(preprocess(text), preprocess(text), background=NO_IDF)
    assert r.tfidf_cosine == pytest.approx(1.0)
    assert r.keyword_coverage == pytest.approx(1.0)
    assert r.missing_keywords == []


def test_unrelated_texts_have_zero_similarity():
    r = analyze_keywords(preprocess("Python developer for machine learning."), preprocess("Professional chef baking bread."), background=NO_IDF)
    assert r.tfidf_cosine == 0.0
    assert r.keyword_coverage == 0.0


def test_idf_pushes_down_common_words():
    job = preprocess("Team player. PyTorch models.")
    bg = Background(100, {"team": 90, "player": 30, "team player": 1, "pytorch": 2, "model": 40, "pytorch model": 1}, "test")
    terms = [k.term for k in analyze_keywords(job, preprocess("x y z"), background=bg).job_keywords]
    assert terms.index("pytorch") < terms.index("team")


def test_repeated_term_ranks_first_and_phrases_replace_their_words():
    job = preprocess("Machine learning intern. Machine learning projects. Docker basics.")
    r = analyze_keywords(job, preprocess("Docker user"), background=NO_IDF)
    terms = [k.term for k in r.job_keywords]
    assert terms[0] == "machine learning"
    assert "machine" not in terms and "learning" not in terms  # covered by the repeated bigram (no corpus)
    assert "docker" in r.matched_keywords and "machine learning" in r.missing_keywords


def test_known_phrase_absorbs_its_words():
    job = preprocess("Data analysis intern. Docker basics.")
    r = analyze_keywords(job, preprocess("Docker"), known_phrases={"data analysis"}, background=NO_IDF)
    terms = [k.term for k in r.job_keywords]
    assert "data analysis" in terms and "data" not in terms and "analysis" not in terms
    assert "docker" in terms  # "docker basic" is incidental, so "docker" stays


def test_gerund_compounds_stay_together():
    r = preprocess("Machine learning projects. Developing applications.")
    assert ["machine", "learning", "project"] in r.segments
    assert ["develop"] in r.segments


def test_verbs_are_not_offered_as_keywords():
    r = analyze_keywords(preprocess("Developing Python applications."), preprocess("Python"), background=NO_IDF)
    assert "develop" not in [k.term for k in r.job_keywords]


def test_synonym_matching():
    job, profile = preprocess("Knowledge of AI, Python."), preprocess("Artificial Intelligence student using Python.")
    plain = analyze_keywords(job, profile, background=NO_IDF)
    with_syn = analyze_keywords(job, profile, synonym_groups=[{"ai", "artificial intelligence"}], background=NO_IDF)
    assert "ai" in plain.missing_keywords
    assert "ai" in with_syn.matched_keywords
    assert next(k for k in with_syn.job_keywords if k.term == "ai").match_type == "synonym"


def test_shipped_reference_corpus_is_sane():
    bg = load_background()
    assert bg.n_documents >= 500
    # generic job-ad words must be far more common than specific technologies
    assert bg.df.get("team", 0) > 10 * bg.df.get("pytorch", 1)
    assert idf("pytorch", bg) > idf("team", bg)
    assert np.isfinite(idf("never-seen-term", bg))


def test_rare_bigram_is_not_a_keyword_but_established_one_is():
    job = preprocess("New York hub. New York hub. Machine learning.")
    bg = Background(100, {"new": 50, "york": 10, "hub": 5, "york hub": 1, "machine learning": 30, "machine": 31, "learning": 32}, "test")
    terms = [k.term for k in analyze_keywords(job, preprocess("x y z"), background=bg).job_keywords]
    assert "york hub" not in terms and "machine learning" in terms
