"""End-to-end tests: the full pipeline including spaCy NER and Sentence-BERT."""

import pytest

from modules.nlp_resources import SAMPLE_DIR
from modules.pipeline import analyze
from modules.semantic_similarity import analyze_semantics, cosine_similarity
from modules.text_extraction import extract_text


@pytest.fixture(scope="module")
def sample_result():
    return analyze(
        (SAMPLE_DIR / "sample_profile.txt").read_text(encoding="utf-8"),
        (SAMPLE_DIR / "sample_job.txt").read_text(encoding="utf-8"),
    )


def test_sample_skill_gap(sample_result):
    gap = sample_result.skill_gap
    assert {"Python", "Machine Learning", "Natural Language Processing", "SQL", "Git"} <= set(gap.matched)
    assert {"TensorFlow", "PyTorch", "AWS"} <= set(gap.missing)


def test_sample_scores_are_sane(sample_result):
    s = sample_result.scores
    assert 0 < s.overall < 100
    assert all(0 <= v <= 100 for v in s.components.values())
    assert s.overall == pytest.approx(sum(s.contributions.values()), abs=0.2)


def test_sample_recommendations_are_honest(sample_result):
    text = " ".join(r.message for r in sample_result.recommendations)
    assert "TensorFlow" in text and "AWS" in text
    assert "do not list skills you have not actually used" in text
    assert sample_result.recommendations[0].priority == "High"


def test_entities_include_rule_based_qualification(sample_result):
    labels = {e.label for e in sample_result.profile_entities}
    assert "QUALIFICATION" in labels


def test_semantic_similarity_orders_related_above_unrelated():
    related = analyze_semantics(["We need a deep learning engineer for computer vision."],
                                ["I trained convolutional neural networks to classify images."], "", "")
    unrelated = analyze_semantics(["We need a deep learning engineer for computer vision."],
                                  ["I enjoy baking sourdough bread on weekends."], "", "")
    assert related.document_similarity > unrelated.document_similarity + 0.2


def test_cosine_similarity_formula():
    import numpy as np

    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


@pytest.mark.parametrize("profile, job", [("", "valid job description with words"), ("too short", "valid job description here ok")])
def test_input_validation(profile, job):
    with pytest.raises(ValueError):
        analyze(profile, job)


def test_text_extraction_txt_and_errors():
    assert extract_text("a.txt", "Python développeur".encode("utf-8")) == "Python développeur"
    assert extract_text("a.txt", "caf\xe9".encode("cp1252")) == "café"
    with pytest.raises(ValueError):
        extract_text("a.docx", b"x")
    with pytest.raises(ValueError):
        extract_text("a.pdf", b"not a pdf")
