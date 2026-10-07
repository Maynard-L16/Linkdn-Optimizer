import pytest

from modules.scoring import WEIGHTS, band_for, combine_scores


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_weighted_sum():
    s = combine_scores(semantic=0.5, skills=1.0, keywords=0.0)
    assert s.overall == pytest.approx(0.4 * 50 + 0.3 * 100 + 0.3 * 0)
    assert s.contributions == {"semantic": 20.0, "skills": 30.0, "keywords": 0.0}


def test_missing_component_weight_is_redistributed():
    s = combine_scores(semantic=0.6, skills=None, keywords=0.6)
    assert s.components["skills"] is None
    assert s.effective_weights["skills"] == 0.0
    assert s.effective_weights["semantic"] == pytest.approx(0.4 / 0.7)
    assert s.overall == pytest.approx(60.0)


def test_values_are_clamped():
    assert combine_scores(1.7, -0.2, 0.5).components == {"semantic": 100.0, "skills": 0.0, "keywords": 50.0}


def test_all_missing_gives_zero():
    assert combine_scores(None, None, None).overall == 0.0


@pytest.mark.parametrize("score, band", [(90, "Strong match"), (60, "Good match"), (40, "Partial match"), (10, "Weak match")])
def test_bands(score, band):
    assert band_for(score) == band
