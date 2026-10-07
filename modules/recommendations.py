"""Step 7 - Gap analysis -> profile strengths and improvement recommendations.

WHAT : Turns the numeric results into plain-language feedback.
WHY  : A score alone does not tell the user what to do next.
HOW  : Transparent IF-THEN rules over the analysis results (no generative AI),
       so every recommendation can be traced back to a detected gap.
ETHICS: Recommendations never tell the user to add a skill they lack. They
       ask the user to make REAL experience visible, or to build it first.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from modules.ner_analysis import Entity
from modules.preprocessing import PreprocessResult
from modules.scoring import ScoreBreakdown
from modules.semantic_similarity import SemanticResult
from modules.skill_extraction import SkillGap, skill_category, synonym_groups
from modules.tfidf_analysis import TfidfResult

SHORT_PROFILE_WORDS = 80
WEAK_REQUIREMENT_SIM = 0.35
STRONG_REQUIREMENT_SIM = 0.60
MAX_REQUIREMENT_ITEMS = 3
NUMBER_RE = re.compile(r"\d")
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


@dataclass
class Recommendation:
    priority: str  # High / Medium / Low
    area: str
    message: str


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _shorten(sentence: str, limit: int = 140) -> str:
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rsplit(" ", 1)[0] + "…"


SOFT_SKILL_CATEGORY = "Professional Skills"


COMPANY_VOICE_RE = re.compile(r"\b(?:we|our|us)\b", re.IGNORECASE)


def _patterns(terms: set[str]) -> list[re.Pattern[str]]:
    return [re.compile(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", re.IGNORECASE) for t in sorted(terms) if len(t) > 1]


def requirement_terms(gap: SkillGap, tfidf: TfidfResult) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    """(technical-skill patterns, keyword patterns) for the job."""
    names = [n for n in gap.matched + gap.missing + list(gap.partial) if skill_category(n) != SOFT_SKILL_CATEGORY]
    tech = {a for g in synonym_groups(names) for a in g}
    keywords = {k.term for k in tfidf.job_keywords}
    return _patterns(tech), _patterns(keywords)


def is_requirement(sentence: str, patterns: tuple[list[re.Pattern[str]], list[re.Pattern[str]]]) -> bool:
    """Is this job sentence a requirement of the role (and not company/culture text)?

    Yes if it names a technical skill the job asks for, or if it contains a top
    keyword and is not written in the company's own voice ("we", "our", "us").
    """
    tech, keywords = patterns
    if any(p.search(sentence) for p in tech):
        return True
    return not COMPANY_VOICE_RE.search(sentence) and any(p.search(sentence) for p in keywords)


def find_strengths(gap: SkillGap, tfidf: TfidfResult, semantic: SemanticResult, scores: ScoreBreakdown) -> list[str]:
    strengths: list[str] = []
    requested = len(gap.matched) + len(gap.partial) + len(gap.missing)
    if gap.matched:
        strengths.append(f"Your profile shows {len(gap.matched)} of the {requested} skills the job asks for: {_join(gap.matched)}.")

    by_category: dict[str, list[str]] = defaultdict(list)
    for name in gap.matched + list(gap.partial) + gap.missing:
        by_category[skill_category(name)].append(name)
    for category, names in sorted(by_category.items()):
        if len(names) >= 2 and all(n in gap.matched for n in names):
            strengths.append(f"Complete coverage of the requested {category} skills ({_join(names)}).")

    patterns = requirement_terms(gap, tfidf)
    strong = [m for m in semantic.requirement_matches if m.similarity >= STRONG_REQUIREMENT_SIM and is_requirement(m.job_sentence, patterns)]
    for m in sorted(strong, key=lambda m: -m.similarity)[:MAX_REQUIREMENT_ITEMS]:
        strengths.append(
            f'The requirement "{_shorten(m.job_sentence)}" is closely matched ({m.similarity:.0%} similar) by: '
            f'"{_shorten(m.best_profile_sentence)}".'
        )

    for key, label in (("semantic", "overall semantic similarity"), ("keywords", "coverage of the job's key terms")):
        value = scores.components.get(key)
        if value is not None and value >= 70:
            strengths.append(f"High {label} ({value:.0f}%).")
    return strengths


def build_recommendations(
    gap: SkillGap,
    tfidf: TfidfResult,
    semantic: SemanticResult,
    scores: ScoreBreakdown,
    profile: PreprocessResult,
    profile_entities: list[Entity],
    job_entities: list[Entity],
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    # 1. Missing skills, grouped by category. Technical and professional
    #    (soft) skills need different advice, so they are separate items.
    missing_by_category: dict[str, list[str]] = defaultdict(list)
    for name in gap.missing:
        missing_by_category[skill_category(name)].append(name)
    soft = missing_by_category.pop(SOFT_SKILL_CATEGORY, [])
    if missing_by_category:
        technical = [n for names in missing_by_category.values() for n in names]
        groups = "; ".join(f"{cat}: {', '.join(names)}" for cat, names in sorted(missing_by_category.items()))
        it = "it" if len(technical) == 1 else "them"
        recs.append(
            Recommendation(
                "High",
                "Missing technical skills",
                f"Requested by the job but not detected in your profile - {groups}. If you have genuinely used "
                f"{it} (projects, coursework, internships, certifications), mention {it} explicitly together with "
                f"what you built. If you have not, consider a small hands-on project or a course first - do not "
                f"list skills you have not actually used.",
            )
        )
    if soft:
        recs.append(
            Recommendation(
                "Medium",
                "Professional skills",
                f"The job emphasises {_join(soft)}. Rather than just listing these words, show them with real "
                "examples - team projects, presentations, hackathons, leadership or volunteering roles you have held.",
            )
        )

    # 2. Partial evidence: the job's exact term is absent but a specific related skill is present.
    for name, evidence in gap.partial.items():
        recs.append(
            Recommendation(
                "Medium",
                "Use the job's exact term",
                f"The job asks for {name}; your profile shows {_join(evidence)}. Keyword filters often search for "
                f"the exact term, so if it is accurate, state it explicitly (for example \"{name} ({evidence[0]})\").",
            )
        )

    # 3. Important job keywords that are missing and are NOT already covered by a missing-skill rec.
    skill_terms = {alias for group in synonym_groups(gap.missing + list(gap.partial)) for alias in group}
    missing_terms = [t for t in tfidf.missing_keywords if t not in skill_terms]
    if missing_terms:
        recs.append(
            Recommendation(
                "Medium",
                "Important job keywords",
                f"Key terms from the job description not found in your profile: {_join([f'“{t}”' for t in missing_terms[:8]])}. "
                "Where they truthfully describe your work, use the job's own wording in your headline, About "
                "section or project descriptions.",
            )
        )

    # 4. Job requirements that no profile sentence addresses well (semantic view).
    patterns = requirement_terms(gap, tfidf)
    weak = sorted(
        (m for m in semantic.requirement_matches if m.similarity < WEAK_REQUIREMENT_SIM and is_requirement(m.job_sentence, patterns)),
        key=lambda m: m.similarity,
    )
    for m in weak[:MAX_REQUIREMENT_ITEMS]:
        recs.append(
            Recommendation(
                "Medium",
                "Weakly addressed requirement",
                f'This part of the job description is not well reflected in your profile ({m.similarity:.0%} best '
                f'sentence similarity): "{_shorten(m.job_sentence)}". If you have related experience, describe it '
                "in a sentence of its own.",
            )
        )

    # 5. Qualifications the job mentions that the profile does not.
    profile_quals = {e.text.lower() for e in profile_entities if e.label == "QUALIFICATION"}
    job_quals = [e.text for e in job_entities if e.label == "QUALIFICATION" and e.text.lower() not in profile_quals]
    if job_quals:
        recs.append(
            Recommendation(
                "Low",
                "Qualifications",
                f"The job mentions {_join([f'“{q}”' for q in job_quals[:5]])}. If you hold these, make sure your "
                "education/certification section states them clearly.",
            )
        )

    # 6. Profile depth.
    word_count = len(profile.original.split())
    if word_count < SHORT_PROFILE_WORDS:
        recs.append(
            Recommendation(
                "Medium",
                "Profile detail",
                f"Your profile text is short ({word_count} words), which gives every analysis less to work with. "
                "For each project, describe the problem, the tools you used and the outcome.",
            )
        )

    # 7. Measurable results.
    if not NUMBER_RE.search(profile.original):
        recs.append(
            Recommendation(
                "Low",
                "Measurable results",
                "No numbers were detected in your profile. Where you have real figures (model accuracy, dataset "
                "size, users, time saved), adding them makes your project descriptions more concrete.",
            )
        )

    # 8. Relevant skills first.
    if gap.extra and gap.matched:
        recs.append(
            Recommendation(
                "Low",
                "Ordering",
                f"Your profile also lists skills this job does not ask for ({_join(gap.extra[:6])}). Keep them, but "
                f"for this application consider leading with the requested ones ({_join(gap.matched[:6])}).",
            )
        )

    if not recs:
        recs.append(
            Recommendation(
                "Low",
                "Fine-tuning",
                f"No major gaps were detected ({scores.band.lower()}). Review your headline and About section so "
                "they mention the role title and your most relevant projects.",
            )
        )
    return sorted(recs, key=lambda r: PRIORITY_ORDER[r.priority])
