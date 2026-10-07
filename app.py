"""Streamlit UI for the NLP-Based LinkedIn Profile Optimizer & Job Matcher.

Run with:   streamlit run app.py

This file only handles input and display. All NLP happens in modules/
(entry point: modules.pipeline.analyze).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.nlp_resources import SAMPLE_DIR, get_nlp, get_stopwords
from modules.pipeline import AnalysisResult, analyze
from modules.scoring import COMPONENT_LABELS, WEIGHTS
from modules.semantic_similarity import MODEL_NAME, get_model
from modules.skill_extraction import PARTIAL_CREDIT
from modules.text_extraction import extract_text

st.set_page_config(page_title="LinkedIn Profile Optimizer & Job Matcher", page_icon="🔎", layout="wide")

PRIORITY_COLORS = {"High": "red", "Medium": "orange", "Low": "blue"}


# --------------------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading NLP models (first run only, ~10-20 s)...")
def load_models() -> str:
    """Load spaCy, the stop-word list and Sentence-BERT once per server process."""
    get_nlp()
    _, stopword_source = get_stopwords()
    get_model()
    return stopword_source


# --------------------------------------------------------------------------- callbacks
def load_sample() -> None:
    st.session_state.profile_text = (SAMPLE_DIR / "sample_profile.txt").read_text(encoding="utf-8")
    st.session_state.job_text = (SAMPLE_DIR / "sample_job.txt").read_text(encoding="utf-8")
    st.session_state.pop("result", None)


def clear_inputs() -> None:
    st.session_state.profile_text = ""
    st.session_state.job_text = ""
    st.session_state.pop("result", None)


def on_upload(kind: str) -> None:
    """Fill the text area from an uploaded PDF/TXT (runs before the page re-renders)."""
    uploaded = st.session_state.get(f"{kind}_file")
    if uploaded is None:
        return
    try:
        st.session_state[f"{kind}_text"] = extract_text(uploaded.name, uploaded.getvalue())
        st.session_state[f"{kind}_msg"] = ("success", f"Extracted text from {uploaded.name}. You can edit it below.")
    except ValueError as exc:
        st.session_state[f"{kind}_msg"] = ("error", str(exc))


# --------------------------------------------------------------------------- small render helpers
def badges(names: list[str], color: str, empty: str = "None") -> None:
    if not names:
        st.caption(empty)
        return
    with st.container(horizontal=True, gap="small"):
        for name in names:
            st.badge(name, color=color)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


# --------------------------------------------------------------------------- sidebar
def render_sidebar(stopword_source: str) -> None:
    with st.sidebar:
        st.header("About")
        st.write(
            "An NLP / Text Analytics project that compares a LinkedIn profile or resume with a job "
            "description using classical and neural NLP techniques - no paid or generative AI APIs."
        )
        st.subheader("Pipeline")
        st.markdown(
            "1. Text extraction (PDF/TXT)\n"
            "2. Preprocessing - tokenize, lowercase, stop-words, lemmatize\n"
            "3. Skill extraction - skill dictionary (`data/skills.json`)\n"
            "4. TF-IDF keyword analysis\n"
            "5. Named Entity Recognition - spaCy\n"
            "6. Semantic embeddings - Sentence-BERT\n"
            "7. Cosine similarity\n"
            "8. Gap analysis -> score + recommendations"
        )
        st.subheader("Score weights")
        for key, weight in WEIGHTS.items():
            st.write(f"- {COMPONENT_LABELS[key]}: **{weight:.0%}**")
        st.subheader("Models")
        st.caption(f"spaCy `en_core_web_sm` · `{MODEL_NAME}` · {stopword_source}")
        st.caption("Privacy: your text is processed in memory on this computer and is not stored or sent anywhere.")


# --------------------------------------------------------------------------- results
def render_scores(r: AnalysisResult) -> None:
    s = r.scores
    st.header("Results")
    with st.container(border=True):
        left, right = st.columns([1, 2], vertical_alignment="center")
        with left:
            st.metric("Overall Match Score", f"{s.overall:.1f}%")
            st.badge(s.band, color="green" if s.overall >= 55 else "orange" if s.overall >= 35 else "red")
        with right:
            st.progress(s.overall / 100)
            terms = " + ".join(
                f"{s.effective_weights[k]:.2f} × {s.components[k]:.1f}" for k in WEIGHTS if s.components[k] is not None
            )
            st.caption(f"Overall = {terms} = **{s.overall:.1f}**")
            st.caption(
                "This is an analytical text-compatibility score. It is NOT a probability of being shortlisted "
                "or hired, and it cannot see experience that is not written in the text."
            )

    cols = st.columns(3)
    help_text = {
        "semantic": "Cosine similarity of Sentence-BERT embeddings (meaning, not exact words).",
        "skills": f"Share of the job's dictionary skills found in the profile (related skill = {PARTIAL_CREDIT} credit).",
        "keywords": "TF-IDF-weighted share of the job's top keywords that appear in the profile.",
    }
    for col, key in zip(cols, WEIGHTS):
        value = s.components[key]
        with col, st.container(border=True):
            st.metric(COMPONENT_LABELS[key], pct(value), help=help_text[key])
            st.progress((value or 0) / 100)
            st.caption(f"Weight {s.effective_weights[key]:.0%} → contributes {s.contributions[key]:.1f} pts")


def render_feedback(r: AnalysisResult) -> None:
    left, right = st.columns([2, 3])
    with left, st.container(border=True):
        st.subheader("Profile strengths")
        if r.strengths:
            for line in r.strengths:
                st.markdown(f"- {line}")
        else:
            st.caption("No strong overlaps detected yet - see the recommendations.")
    with right, st.container(border=True):
        st.subheader("Improvement recommendations")
        for rec in r.recommendations:
            st.badge(f"{rec.priority} · {rec.area}", color=PRIORITY_COLORS[rec.priority])
            st.write(rec.message)
        st.caption("Rule-based suggestions generated from the detected gaps. Only add what is true about you.")


def render_skills_tab(r: AnalysisResult) -> None:
    gap = r.skill_gap
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**Matched skills ({len(gap.matched)})**")
        badges(gap.matched, "green")
    with c2:
        st.markdown(f"**Missing skills ({len(gap.missing)})**")
        badges(gap.missing, "red")
    with c3:
        st.markdown(f"**Partially evidenced ({len(gap.partial)})**")
        badges([f"{k} ← {', '.join(v)}" for k, v in gap.partial.items()], "orange")
    with c4:
        st.markdown(f"**Additional profile skills ({len(gap.extra)})**")
        badges(gap.extra, "gray")

    rows = []
    for name in sorted(set(r.job_skills) | set(r.profile_skills)):
        match = r.job_skills.get(name) or r.profile_skills[name]
        status = (
            "Matched" if name in gap.matched else "Partial" if name in gap.partial
            else "Missing" if name in gap.missing else "Profile only"
        )
        rows.append({
            "Skill": name,
            "Category": match.category,
            "Status": status,
            "Mentions in job": r.job_skills[name].count if name in r.job_skills else 0,
            "Mentions in profile": r.profile_skills[name].count if name in r.profile_skills else 0,
            "Text matched": ", ".join(sorted(set().union(*(d[name].surface_forms for d in (r.job_skills, r.profile_skills) if name in d)))),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Skill coverage = (matched + {PARTIAL_CREDIT} × partial) / skills requested by the job. "
        "'Partial' means the job's exact skill is absent but a more specific related skill is present "
        "(e.g. job: SQL, profile: MySQL). Skills come from a dictionary of aliases in data/skills.json."
    )


def render_keywords_tab(r: AnalysisResult) -> None:
    t = r.tfidf
    c1, c2 = st.columns(2)
    c1.metric("Keyword coverage (used in score)", pct(None if t.keyword_coverage is None else 100 * t.keyword_coverage))
    c2.metric("TF-IDF cosine similarity (reference)", f"{100 * t.tfidf_cosine:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Important job keywords (TF-IDF)**")
        df = pd.DataFrame([
            {"Keyword": k.term, "Count in job": k.count, "IDF": round(k.idf, 2), "TF-IDF weight": round(k.weight, 3),
             "In profile": {"exact": "✅ exact", "synonym": "✅ synonym"}.get(k.match_type, "❌ missing")}
            for k in t.job_keywords
        ])
        st.dataframe(df, hide_index=True)
        st.markdown("**Missing keywords**")
        badges(t.missing_keywords, "red", "None - every top keyword is present.")
    with right:
        st.markdown("**Job vs profile TF-IDF weights (shared vector space)**")
        if t.comparison:
            chart = pd.DataFrame(t.comparison).rename(columns={"job_weight": "Job", "profile_weight": "Profile"})
            st.bar_chart(chart, x="term", y=["Job", "Profile"], horizontal=True, stack=False,
                         x_label="TF-IDF weight", y_label="", height=max(300, 22 * len(chart)))
    st.caption(
        "Terms = unigrams and noun-phrase bigrams of the lemmatised text. tf = 1 + ln(count); "
        "idf = ln((1 + N) / (1 + df)) + 1, with document frequencies from a reference corpus of "
        f"{t.idf_source}. Common job-ad words get a low IDF, specific skills a high IDF. "
        "Keyword coverage = sum of weights of found keywords / sum of weights of all top keywords. "
        "'Synonym' = matched through skill aliases (e.g. 'AI' ↔ 'Artificial Intelligence')."
    )


def render_semantic_tab(r: AnalysisResult) -> None:
    sem = r.semantic
    st.metric("Document-level cosine similarity", f"{100 * sem.document_similarity:.1f}%")
    st.markdown("**Best-matching profile sentence for every job sentence**")
    df = pd.DataFrame([
        {"Job requirement": m.job_sentence, "Best matching profile sentence": m.best_profile_sentence,
         "Similarity": round(100 * m.similarity, 1)}
        for m in sem.requirement_matches
    ])
    st.dataframe(df, hide_index=True, column_config={
        "Similarity": st.column_config.ProgressColumn("Similarity", format="%.1f%%", min_value=0, max_value=100),
    })
    st.caption(
        f"Each sentence is encoded by the pretrained {sem.model_name} model into a 384-dimensional vector. "
        "The document vector is the mean of its sentence vectors; similarity = cosine of the angle between "
        "vectors. Embeddings capture meaning, so paraphrases score high even without shared words."
    )


def render_entities_tab(r: AnalysisResult) -> None:
    left, right = st.columns(2)
    for col, title, ents in ((left, "Profile entities", r.profile_entities), (right, "Job description entities", r.job_entities)):
        with col:
            st.markdown(f"**{title}**")
            if ents:
                st.dataframe(pd.DataFrame([
                    {"Entity": e.text, "Label": e.label, "Meaning": e.description, "Source": e.source, "Count": e.count}
                    for e in ents
                ]), hide_index=True)
            else:
                st.caption("No entities found.")
    st.markdown("**How does the generic NER model label the skills our dictionary found?**")
    st.dataframe(pd.DataFrame([
        {"Skill (dictionary)": c.skill, "Text": c.text_in_document, "spaCy NER label": c.generic_ner_label}
        for c in r.ner_skill_check
    ]), hide_index=True)
    st.info(
        "spaCy's en_core_web_sm NER was trained on news and web text (OntoNotes 5), not resumes. It often "
        "labels technologies as organisations, places or people - or misses them. That is why skills are "
        "extracted with a dedicated dictionary, while NER is used for organisations, locations, dates and "
        "(through a rule-based EntityRuler) qualifications."
    )


def render_internals_tab(r: AnalysisResult, stopword_source: str) -> None:
    st.caption(f"Stop-word list: {stopword_source}")
    for title, p in (("Profile", r.profile), ("Job description", r.job)):
        with st.expander(f"{title} - preprocessing steps", expanded=False):
            steps = [
                ("1. Noise cleaning (URLs, e-mails, phones, bullets)", p.cleaned),
                ("2-3. Tokenization + lowercasing", p.tokens),
                ("4. Punctuation / number / noise removal", p.tokens_no_noise),
                ("5. Stop-word removal", p.tokens_no_stopwords),
                ("6. Lemmatization (final tokens)", p.lemmas),
            ]
            for label, value in steps:
                if isinstance(value, str):
                    st.markdown(f"**{label}** - {len(value)} characters")
                    st.code(value[:1500] + ("…" if len(value) > 1500 else ""), language=None, wrap_lines=True)
                else:
                    st.markdown(f"**{label}** - {len(value)} tokens")
                    st.code(" | ".join(value[:150]) + (" …" if len(value) > 150 else ""), language=None, wrap_lines=True)
            st.markdown(f"**Sentences for embeddings** - {len(p.sentences)}")
            st.write(p.sentences[:20])
    removed = r.job.removed_boilerplate
    with st.expander(f"Job description - {len(removed)} boilerplate sentences removed before analysis"):
        st.caption("Legal, benefits, privacy and hiring-process sentences (cue phrases in data/jd_boilerplate_cues.txt) "
                   "are not requirements of the role, so they are excluded from keywords, skills and similarity.")
        st.write(removed or "None found.")
    st.markdown("**Stage timings**")
    st.dataframe(pd.DataFrame([{"Stage": k, "Time (ms)": v} for k, v in r.timings_ms.items()]), hide_index=True)


# --------------------------------------------------------------------------- main
def main() -> None:
    try:
        stopword_source = load_models()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(stopword_source)
    st.title("LinkedIn Profile Optimizer & Job Matcher")
    st.write(
        "Paste or upload your LinkedIn profile / resume and a target job description. The NLP pipeline measures "
        "how well they match and shows which skills and keywords are missing."
    )
    with st.container(horizontal=True):
        st.button("Load sample data", on_click=load_sample)
        st.button("Clear", on_click=clear_inputs, type="tertiary")

    st.session_state.setdefault("profile_text", "")
    st.session_state.setdefault("job_text", "")
    left, right = st.columns(2)
    for col, kind, title, types in (
        (left, "profile", "1 · Profile / Resume", ["pdf", "txt"]),
        (right, "job", "2 · Job Description", ["pdf", "txt"]),
    ):
        with col, st.container(border=True):
            st.subheader(title)
            st.file_uploader(f"Optional: upload {'PDF or TXT' if kind == 'job' else 'a PDF (e.g. LinkedIn → More → Save to PDF) or TXT'}",
                             type=types, key=f"{kind}_file", on_change=on_upload, args=(kind,))
            if msg := st.session_state.pop(f"{kind}_msg", None):
                (st.success if msg[0] == "success" else st.error)(msg[1])
            st.text_area(
                "Text", key=f"{kind}_text", height=260, label_visibility="collapsed",
                placeholder="Paste your headline, About section, experience, projects and skills..." if kind == "profile"
                else "Paste the full job description...",
            )
            st.caption(f"{len(st.session_state[f'{kind}_text'].split())} words")

    if st.button("3 · Analyze profile against job", type="primary", width="stretch"):
        try:
            with st.spinner("Running the NLP pipeline..."):
                st.session_state.result = analyze(st.session_state.profile_text, st.session_state.job_text)
        except ValueError as exc:
            st.session_state.pop("result", None)
            st.warning(str(exc))

    result: AnalysisResult | None = st.session_state.get("result")
    if result is None:
        return

    render_scores(result)
    render_feedback(result)
    tabs = st.tabs(["Skills gap", "Keywords (TF-IDF)", "Semantic similarity", "Entities (NER)", "Pipeline internals"])
    with tabs[0]:
        render_skills_tab(result)
    with tabs[1]:
        render_keywords_tab(result)
    with tabs[2]:
        render_semantic_tab(result)
    with tabs[3]:
        render_entities_tab(result)
    with tabs[4]:
        render_internals_tab(result, stopword_source)


main()
