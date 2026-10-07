"""Step 4 - Named Entity Recognition (NER) with spaCy.

WHAT : Finds named entities - organisations, places, dates, products - and,
       via our rule-based EntityRuler, qualifications (degrees, certifications).
WHY  : Entities give structured context a recruiter cares about: where the
       person studied/worked, where the job is, what degree is asked for.
MODEL: en_core_web_sm - a statistical (CNN-based) NER trained on the
       OntoNotes 5 corpus (news, web, conversation text) + our EntityRuler.
INPUT: the spaCy Doc already produced in preprocessing (no second parse).
OUTPUT: list of Entity rows, plus a comparison showing how the GENERIC model
       labels the skills our dictionary found.

Known limitation (shown in the UI on purpose): OntoNotes contains almost no
resumes, so the generic model often mislabels technical skills or misses
them. Observed with en_core_web_sm on the sample data: "Python" and "C++" ->
GPE (countries/cities), "Git" -> PERSON, "TensorFlow", "PyTorch", "Node.js" ->
ORG. That is exactly why skill extraction uses a dictionary instead of NER.
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy
from spacy.tokens import Doc

from modules.skill_extraction import SkillMatch

# Numeric labels add noise for this task ("two", "3", "first").
IGNORED_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"}
LABEL_DESCRIPTIONS = {"QUALIFICATION": "Degree, certification or academic qualification (rule-based)"}


@dataclass
class Entity:
    text: str
    label: str
    description: str
    source: str  # "statistical model" or "rule-based (EntityRuler)"
    count: int = 1


@dataclass
class SkillNerCheck:
    skill: str
    text_in_document: str
    generic_ner_label: str  # what spaCy's statistical NER called it


def describe(label: str) -> str:
    return LABEL_DESCRIPTIONS.get(label) or spacy.explain(label) or label


def extract_entities(doc: Doc) -> list[Entity]:
    """Unique (text, label) entities with mention counts, most frequent first."""
    found: dict[tuple[str, str], Entity] = {}
    for ent in doc.ents:
        text = " ".join(ent.text.split())
        if ent.label_ in IGNORED_LABELS or len(text) < 2:
            continue
        key = (text.lower(), ent.label_)
        if key in found:
            found[key].count += 1
        else:
            source = "rule-based (EntityRuler)" if ent.ent_id_ == "rule_based" else "statistical model"
            found[key] = Entity(text, ent.label_, describe(ent.label_), source)
    return sorted(found.values(), key=lambda e: (e.label, -e.count, e.text.lower()))


def compare_ner_with_skills(doc: Doc, skills: dict[str, SkillMatch]) -> list[SkillNerCheck]:
    """For every dictionary skill mention, report the label the generic NER gave that span."""
    rows: list[SkillNerCheck] = []
    for name, match in skills.items():
        start, end = match.spans[0]
        label = "not recognised"
        for ent in doc.ents:
            if ent.start_char < end and ent.end_char > start and ent.ent_id_ != "rule_based":
                label = f"{ent.label_} ({describe(ent.label_)})"
                break
        rows.append(SkillNerCheck(name, doc.text[start:end], label))
    return rows
