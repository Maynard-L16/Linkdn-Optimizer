import pytest

from modules.preprocessing import clean_text, preprocess


def test_clean_text_removes_noise():
    raw = "Visit https://github.com/me or www.me.dev\nMail: a.b@x.com Phone +91 98765 43210\n• Python\nPage 1 of 3"
    cleaned = clean_text(raw)
    for noise in ("https", "www", "@", "98765", "Page 1 of 3", "•"):
        assert noise not in cleaned
    assert "Python" in cleaned


def test_clean_text_joins_pdf_line_wraps():
    assert clean_text("Developed a model for\nstock prediction") == "Developed a model for stock prediction"
    assert clean_text("Skills\nPython") == "Skills\nPython"  # capitalised line = real new line


def test_each_step_on_sample_sentence():
    r = preprocess("We are developing Machine Learning applications, using Python 3!")
    assert r.tokens[:4] == ["we", "are", "developing", "machine"]  # lowercased tokens
    assert "," not in r.tokens_no_noise and "3" not in r.tokens_no_noise  # punctuation/numbers gone
    assert "we" not in r.tokens_no_stopwords and "are" not in r.tokens_no_stopwords  # stop-words gone
    assert r.lemmas == ["develop", "machine", "learning", "application", "python"]


def test_tech_tokens_survive():
    r = preprocess("Built REST APIs with Node.js, C++ and C# on AWS.")
    for token in ("node.js", "c++", "c#", "aws"):
        assert token in r.lemmas


def test_data_is_not_lemmatised_to_datum():
    assert "data" in preprocess("Strong data analysis skills.").lemmas


def test_segments_do_not_cross_punctuation_or_verbs():
    r = preprocess("Python, machine learning and developing applications.")
    assert ["machine", "learning"] in r.segments
    assert ["python", "machine"] not in r.segments
    assert "develop" in r.non_phrase_lemmas


def test_bullets_and_lines_become_sentences():
    r = preprocess("Projects\n• Built a chatbot using NLP\n• Trained a CNN for images")
    assert r.sentences == ["Projects", "Built a chatbot using NLP", "Trained a CNN for images"]


@pytest.mark.parametrize("text", ["", "   ", "!!! ... ???"])
def test_empty_or_punctuation_only_input(text):
    r = preprocess(text)
    assert r.lemmas == []


def test_boilerplate_sentences_are_removed_from_job_text():
    from modules.preprocessing import remove_boilerplate

    text = "Requirements: Python and SQL.\nWe are an equal opportunity employer. Benefits include dental.\nStrong communication."
    kept, removed = remove_boilerplate(text)
    assert "Python and SQL" in kept and "Strong communication" in kept
    assert removed == ["We are an equal opportunity employer.", "Benefits include dental."]
    r = preprocess(text, drop_boilerplate=True)
    assert r.removed_boilerplate == removed and "dental" not in r.lemmas
