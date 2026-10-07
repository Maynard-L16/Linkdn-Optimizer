"""Step 2 - Dictionary-based skill extraction and skill gap comparison.

WHAT : Finds known skills (from data/skills.json) in a text, then compares the
       skills of the profile against the skills the job asks for.
WHY  : A pretrained NER model does not know that "PyTorch" is a skill (it may
       call it an ORG or miss it). A curated dictionary of skills + aliases
       ("ml" -> Machine Learning, "k8s" -> Kubernetes) is precise, explainable
       and easy to extend - the standard "gazetteer" approach in information
       extraction.
INPUT: cleaned text (original case is kept so case-sensitive aliases work).
OUTPUT: dict  canonical skill name -> SkillMatch (category, mention count,
       surface forms found, character spans), and a SkillGap comparison.

Matching rules:
  * case-insensitive, except aliases listed under "case_sensitive"
    ("Excel" the tool vs "excel" the verb)
  * custom word boundaries so "Java" does not match inside "JavaScript" and
    "SQL" does not match inside "MySQL", but "C++" and "Node.js" still work
  * spaces in an alias also match hyphens ("machine-learning")
  * longest match wins: "React Native" is not also counted as "React"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from modules.nlp_resources import SKILLS_PATH

# Partial credit given in skill coverage when a requested skill is not named
# but a MORE SPECIFIC related skill is (job: "SQL", profile: "MySQL").
PARTIAL_CREDIT = 0.5

# A character that may NOT appear directly before/after a match.
_BOUNDARY = r"A-Za-z0-9+#&"  # "&" so the "R" in "R&D" is not the R language


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...]
    case_sensitive: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


@dataclass
class SkillMatch:
    name: str
    category: str
    count: int = 0
    surface_forms: set[str] = field(default_factory=set)
    spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SkillGap:
    matched: list[str]  # in job AND profile
    partial: dict[str, list[str]]  # job skill -> related profile skills that are evidence for it
    missing: list[str]  # in job, no evidence in profile
    extra: list[str]  # in profile only (not requested)
    coverage: float | None  # 0..1, None when the job lists no known skills


@lru_cache(maxsize=4)
def load_skill_db(path: Path = SKILLS_PATH) -> tuple[Skill, ...]:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    skills = []
    for category, items in raw["categories"].items():
        for item in items:
            skills.append(
                Skill(
                    name=item["name"],
                    category=category,
                    aliases=tuple(a.lower() for a in item.get("aliases", [])),
                    case_sensitive=tuple(item.get("case_sensitive", [])),
                    related=tuple(item.get("related", [])),
                )
            )
    names = [s.name for s in skills]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate skill names in {path}: {sorted(duplicates)}")
    return tuple(skills)


def _alias_pattern(alias: str, case_sensitive: bool) -> re.Pattern[str]:
    body = re.escape(alias).replace(r"\ ", r"[\s\-_]+")
    # Allow a simple plural ("neural networks", "rest apis") for normal words.
    plural = "(?:e?s)?" if alias[-1].isalpha() and len(alias) > 2 and not case_sensitive else ""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![{_BOUNDARY}]){body}{plural}(?![{_BOUNDARY}])", flags)


@lru_cache(maxsize=4)
def _compiled_patterns(path: Path = SKILLS_PATH) -> tuple[tuple[Skill, re.Pattern[str]], ...]:
    patterns = []
    for skill in load_skill_db(path):
        for alias in skill.aliases:
            patterns.append((skill, _alias_pattern(alias, case_sensitive=False)))
        for alias in skill.case_sensitive:
            patterns.append((skill, _alias_pattern(alias, case_sensitive=True)))
    return tuple(patterns)


def extract_skills(text: str, path: Path = SKILLS_PATH) -> dict[str, SkillMatch]:
    """Return every dictionary skill mentioned in `text`, keyed by canonical name."""
    candidates: list[tuple[int, int, Skill]] = []
    for skill, pattern in _compiled_patterns(path):
        for m in pattern.finditer(text):
            candidates.append((m.start(), m.end(), skill))

    # Longest-match-first: resolve overlaps such as "React Native" vs "React".
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    taken: list[tuple[int, int]] = []
    found: dict[str, SkillMatch] = {}
    for start, end, skill in candidates:
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        taken.append((start, end))
        match = found.setdefault(skill.name, SkillMatch(skill.name, skill.category))
        match.count += 1
        match.surface_forms.add(text[start:end])
        match.spans.append((start, end))
    return dict(sorted(found.items()))


def compare_skills(
    profile_skills: dict[str, SkillMatch],
    job_skills: dict[str, SkillMatch],
    path: Path = SKILLS_PATH,
) -> SkillGap:
    """Set comparison of job skills vs profile skills, with partial credit for related skills."""
    db = {s.name: s for s in load_skill_db(path)}
    job_set, profile_set = set(job_skills), set(profile_skills)

    matched = sorted(job_set & profile_set)
    partial: dict[str, list[str]] = {}
    missing: list[str] = []
    for name in sorted(job_set - profile_set):
        evidence = sorted(r for r in db[name].related if r in profile_set)
        if evidence:
            partial[name] = evidence
        else:
            missing.append(name)
    extra = sorted(profile_set - job_set)

    coverage = None
    if job_set:
        coverage = (len(matched) + PARTIAL_CREDIT * len(partial)) / len(job_set)
    return SkillGap(matched=matched, partial=partial, missing=missing, extra=extra, coverage=coverage)


def synonym_groups(skill_names: list[str] | set[str], path: Path = SKILLS_PATH) -> list[set[str]]:
    """Lower-cased alias sets for the given skills (used for synonym-aware keyword matching)."""
    db = {s.name: s for s in load_skill_db(path)}
    return [
        {a.lower() for a in (*db[n].aliases, *db[n].case_sensitive)} | {n.lower()} for n in skill_names if n in db
    ]


def detect_employer_skills(job_text: str, job_skills: dict[str, SkillMatch]) -> list[str]:
    """Skills that are probably the EMPLOYER's name, e.g. "Figma" in Figma's own
    job ad ("At Figma, ...", "Figma's mission"). Heuristic: the matched text is
    used after "at"/"join" or in the possessive form.
    """
    employer = []
    for name, match in job_skills.items():
        for form in match.surface_forms:
            f = re.escape(form)
            if re.search(rf"\b(?:at|join|joining)\s+{f}\b|\b{f}['’]s\b", job_text):
                employer.append(name)
                break
    return sorted(employer)


def known_phrases(path: Path = SKILLS_PATH) -> set[str]:
    """All multi-word aliases ("machine learning", "data analysis")."""
    return {a for s in load_skill_db(path) for a in s.aliases if " " in a}


def skill_category(name: str, path: Path = SKILLS_PATH) -> str:
    for skill in load_skill_db(path):
        if skill.name == name:
            return skill.category
    return "Other"
