import pytest

from modules.skill_extraction import PARTIAL_CREDIT, compare_skills, extract_skills, load_skill_db, synonym_groups


def test_skill_db_loads_and_related_skills_exist():
    db = load_skill_db()
    names = {s.name for s in db}
    assert len(db) >= 140
    for skill in db:
        for related in skill.related:
            assert related in names, f"{skill.name} -> unknown related skill {related}"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Experienced in ML and NLP", {"Machine Learning", "Natural Language Processing"}),
        ("machine-learning engineer", {"Machine Learning"}),
        ("Deployed on k8s with Docker", {"Kubernetes", "Docker"}),
        ("C++, C# and Node.js", {"C++", "C#", "Node.js"}),
        ("built neural networks and REST APIs", {"Neural Networks", "REST APIs"}),
    ],
)
def test_aliases_and_special_characters(text, expected):
    assert set(extract_skills(text)) == expected


@pytest.mark.parametrize(
    "text, absent",
    [
        ("JavaScript developer", "Java"),  # word boundary
        ("MySQL administrator", "SQL"),  # word boundary
        ("I excel at teamwork", "Excel"),  # case-sensitive alias
        ("Our R&D department", "R"),  # & boundary
        ("React Native apps", "React"),  # longest match wins
    ],
)
def test_no_false_positives(text, absent):
    assert absent not in extract_skills(text)


def test_mentions_are_counted():
    found = extract_skills("Python here, python there, Python3 everywhere")
    assert found["Python"].count == 3


def test_compare_skills_sample_case():
    profile = extract_skills("Python, machine learning, NLP, SQL, MongoDB, React, Node.js and Git")
    job = extract_skills("Python, machine learning, deep learning, NLP, SQL, TensorFlow, PyTorch, Git and AWS")
    gap = compare_skills(profile, job)
    assert set(gap.matched) == {"Python", "Machine Learning", "Natural Language Processing", "SQL", "Git"}
    assert set(gap.missing) == {"Deep Learning", "TensorFlow", "PyTorch", "AWS"}
    assert set(gap.extra) == {"MongoDB", "React", "Node.js"}
    assert gap.coverage == pytest.approx(5 / 9)


def test_partial_credit_for_more_specific_skill():
    gap = compare_skills(extract_skills("Worked with MySQL"), extract_skills("Must know SQL and Docker"))
    assert gap.partial == {"SQL": ["MySQL"]}
    assert gap.missing == ["Docker"]
    assert gap.coverage == pytest.approx(PARTIAL_CREDIT / 2)


def test_no_job_skills_gives_none_coverage():
    assert compare_skills(extract_skills("Python"), extract_skills("We need a friendly person")).coverage is None


def test_synonym_groups_include_aliases():
    (group,) = synonym_groups(["Artificial Intelligence"])
    assert {"ai", "artificial intelligence"} <= group
