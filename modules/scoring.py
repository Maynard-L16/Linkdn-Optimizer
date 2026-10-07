"""Step 6 - Transparent combined match score.

WHAT : Combines three independent signals into one 0-100 score.
WHY  : Each signal alone is misleading:
       * semantic similarity rewards "sounding similar" even without the
         required tools,
       * skill coverage ignores everything outside the skill dictionary,
       * keyword coverage misses synonyms and paraphrases.
       A weighted sum of all three is more robust and stays explainable
       (every point of the final score can be traced to one component).

Weights (they sum to 1):
    40 % Semantic similarity  - the broadest view of overall relevance
    30 % Skill coverage       - the most concrete, recruiter-relevant evidence
    30 % Keyword coverage     - ATS-style term overlap, catches non-skill terms
                                 (domain, role and responsibility words)

If a component cannot be computed (e.g. the job description names no skill
from the dictionary), its weight is redistributed proportionally over the
remaining components instead of silently counting it as 0 %.

IMPORTANT: this is an analytical text-compatibility score, NOT a probability
of being shortlisted or hired.
"""

from __future__ import annotations

from dataclasses import dataclass

WEIGHTS: dict[str, float] = {"semantic": 0.40, "skills": 0.30, "keywords": 0.30}
COMPONENT_LABELS = {"semantic": "Semantic Similarity", "skills": "Skill Coverage", "keywords": "Keyword Coverage (TF-IDF)"}

# Interpretation bands for the overall score.
BANDS = [(75, "Strong match"), (55, "Good match"), (35, "Partial match"), (0, "Weak match")]


@dataclass
class ScoreBreakdown:
    overall: float  # 0..100
    band: str
    components: dict[str, float | None]  # 0..100 or None if not computable
    effective_weights: dict[str, float]  # after redistribution
    contributions: dict[str, float]  # points each component adds to `overall`


def band_for(score: float) -> str:
    return next(label for threshold, label in BANDS if score >= threshold)


def combine_scores(semantic: float | None, skills: float | None, keywords: float | None) -> ScoreBreakdown:
    """Inputs are fractions in [0, 1] (or None). Output is on a 0-100 scale."""
    raw = {"semantic": semantic, "skills": skills, "keywords": keywords}
    components = {k: None if v is None else round(100 * min(1.0, max(0.0, v)), 1) for k, v in raw.items()}

    available = {k: WEIGHTS[k] for k, v in components.items() if v is not None}
    total_weight = sum(available.values())
    if total_weight == 0:
        return ScoreBreakdown(0.0, band_for(0.0), components, {k: 0.0 for k in WEIGHTS}, {k: 0.0 for k in WEIGHTS})

    effective = {k: available.get(k, 0.0) / total_weight for k in WEIGHTS}
    contributions = {k: round(effective[k] * (components[k] or 0.0), 1) for k in WEIGHTS}
    overall = round(sum(effective[k] * (components[k] or 0.0) for k in WEIGHTS), 1)
    return ScoreBreakdown(overall, band_for(overall), components, effective, contributions)
