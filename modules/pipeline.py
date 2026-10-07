"""The complete NLP pipeline in one function: analyze(profile_text, job_text).

    Profile + Job text
      -> 1. preprocessing        (clean, tokenize, lowercase, stop-words, lemmatize)
      -> 2. skill extraction     (dictionary matching + gap comparison)
      -> 3. TF-IDF analysis      (important job keywords, keyword coverage, cosine)
      -> 4. NER                  (spaCy entities + rule-based qualifications)
      -> 5. semantic similarity  (Sentence-BERT embeddings + cosine)
      -> 6. scoring              (weighted, transparent combination)
      -> 7. recommendations      (rule-based gap analysis)

The Streamlit UI (app.py) only calls analyze() and displays the result, so
the NLP logic can be tested and demonstrated without the UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from modules.ner_analysis import Entity, SkillNerCheck, compare_ner_with_skills, extract_entities
from modules.preprocessing import PreprocessResult, preprocess
from modules.recommendations import Recommendation, build_recommendations, find_strengths
from modules.scoring import ScoreBreakdown, combine_scores
from modules.semantic_similarity import SemanticResult, analyze_semantics
from modules.skill_extraction import SkillGap, SkillMatch, compare_skills, detect_employer_skills, extract_skills, known_phrases, synonym_groups
from modules.tfidf_analysis import TfidfResult, analyze_keywords

MIN_WORDS = 5
MAX_CHARS = 50_000  # ~ a 15-page resume; protects the UI from accidental huge pastes


@dataclass
class AnalysisResult:
    profile: PreprocessResult
    job: PreprocessResult
    profile_skills: dict[str, SkillMatch]
    job_skills: dict[str, SkillMatch]
    skill_gap: SkillGap
    tfidf: TfidfResult
    profile_entities: list[Entity]
    job_entities: list[Entity]
    ner_skill_check: list[SkillNerCheck]
    semantic: SemanticResult
    scores: ScoreBreakdown
    strengths: list[str]
    recommendations: list[Recommendation]
    employer_skills: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def validate_input(text: str, label: str) -> str:
    text = (text or "").strip()
    if len(text.split()) < MIN_WORDS:
        raise ValueError(f"The {label} is empty or too short (need at least {MIN_WORDS} words).")
    if len(text) > MAX_CHARS:
        raise ValueError(f"The {label} is too long ({len(text):,} characters; limit {MAX_CHARS:,}).")
    return text


def analyze(profile_text: str, job_text: str) -> AnalysisResult:
    profile_text = validate_input(profile_text, "profile / resume text")
    job_text = validate_input(job_text, "job description")
    timings: dict[str, float] = {}

    def timed(stage: str, fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        timings[stage] = round((time.perf_counter() - start) * 1000, 1)
        return out

    profile = timed("1. Preprocessing (profile)", preprocess, profile_text)
    job = timed("1. Preprocessing (job)", preprocess, job_text, drop_boilerplate=True)
    if len(job.cleaned.split()) < MIN_WORDS:
        raise ValueError("After removing legal/benefits boilerplate, too little of the job description is left to analyse.")

    profile_skills = timed("2. Skill extraction (profile)", extract_skills, profile.cleaned)
    job_skills = timed("2. Skill extraction (job)", extract_skills, job.cleaned)
    # "Figma" in Figma's own job ad is the employer, not a required skill.
    employer_skills = detect_employer_skills(job.cleaned, job_skills)
    employer_terms = {a for g in synonym_groups(employer_skills) for a in g}
    job_skills = {k: v for k, v in job_skills.items() if k not in employer_skills}
    gap = compare_skills(profile_skills, job_skills)

    tfidf = timed(
        "3. TF-IDF analysis", analyze_keywords, job, profile,
        synonym_groups=synonym_groups(list(profile_skills)), known_phrases=known_phrases(),
        exclude_terms=employer_terms,
    )

    profile_entities = timed("4. NER (profile)", extract_entities, profile.doc)
    job_entities = timed("4. NER (job)", extract_entities, job.doc)
    ner_check = compare_ner_with_skills(profile.doc, profile_skills) + compare_ner_with_skills(job.doc, {
        k: v for k, v in job_skills.items() if k not in profile_skills
    })

    semantic = timed("5. Semantic similarity", analyze_semantics, job.sentences, profile.sentences, job.cleaned, profile.cleaned)

    scores = combine_scores(semantic=semantic.document_similarity, skills=gap.coverage, keywords=tfidf.keyword_coverage)

    strengths = find_strengths(gap, tfidf, semantic, scores)
    recommendations = build_recommendations(gap, tfidf, semantic, scores, profile, profile_entities, job_entities)

    return AnalysisResult(
        profile=profile,
        job=job,
        profile_skills=profile_skills,
        job_skills=job_skills,
        skill_gap=gap,
        tfidf=tfidf,
        profile_entities=profile_entities,
        job_entities=job_entities,
        ner_skill_check=ner_check,
        semantic=semantic,
        scores=scores,
        strengths=strengths,
        recommendations=recommendations,
        employer_skills=employer_skills,
        timings_ms=timings,
    )


if __name__ == "__main__":  # quick command-line demo: python -m modules.pipeline
    from modules.nlp_resources import SAMPLE_DIR

    result = analyze(
        (SAMPLE_DIR / "sample_profile.txt").read_text(encoding="utf-8"),
        (SAMPLE_DIR / "sample_job.txt").read_text(encoding="utf-8"),
    )
    s = result.scores
    print(f"Overall match: {s.overall}% ({s.band})")
    for key, value in s.components.items():
        print(f"  {key:9s} {value}%  x weight {s.effective_weights[key]:.2f} = {s.contributions[key]} pts")
    print("Matched skills :", result.skill_gap.matched)
    print("Partial skills :", result.skill_gap.partial)
    print("Missing skills :", result.skill_gap.missing)
    print("Job keywords   :", [(k.term, round(k.weight, 3), k.match_type or "missing") for k in result.tfidf.job_keywords])
    print("Entities       :", [(e.text, e.label) for e in result.profile_entities + result.job_entities])
    print("Strengths:")
    for line in result.strengths:
        print("  +", line)
    print("Recommendations:")
    for r in result.recommendations:
        print(f"  [{r.priority}] {r.area}: {r.message}")
    print("Timings (ms):", result.timings_ms)
