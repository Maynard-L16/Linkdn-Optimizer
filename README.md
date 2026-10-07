# NLP-Based LinkedIn Profile Optimizer & Job Matcher

An NLP / Text Analytics application that compares a LinkedIn profile or resume with a target job
description. It measures how well they match, lists matching and missing skills and keywords, and gives
honest, specific recommendations for improving the profile.

All core NLP runs **locally** with classical and pretrained models: spaCy, NLTK, TF-IDF and Sentence-BERT.
There are **no paid or generative-AI APIs**, and nothing is trained from scratch.

---

## 1. Project structure

```text
aveel/
├── app.py                        Streamlit UI (input + dashboard only; no NLP logic)
├── download_models.py            one-time download of spaCy / NLTK / Sentence-BERT resources
├── requirements.txt
├── pytest.ini
├── README.md
│
├── modules/                      the NLP pipeline, one module per technique
│   ├── pipeline.py               analyze(profile, job): runs every stage in order
│   ├── text_extraction.py        step 0: PDF / TXT -> text
│   ├── preprocessing.py          step 1: boilerplate + noise removal, tokenization, lowercasing,
│   │                                     stop-words, lemmatization, phrase segments
│   ├── skill_extraction.py       step 2: dictionary skill extraction + skill-gap comparison
│   ├── tfidf_analysis.py         step 3: TF-IDF keywords, keyword coverage, TF-IDF cosine
│   ├── ner_analysis.py           step 4: spaCy NER + rule-based QUALIFICATION entities
│   ├── semantic_similarity.py    step 5: Sentence-BERT embeddings + cosine similarity
│   ├── scoring.py                step 6: transparent weighted match score
│   ├── recommendations.py        step 7: strengths + rule-based recommendations
│   └── nlp_resources.py          shared paths, cached spaCy pipeline, stop-word list
│
├── data/
│   ├── skills.json               140 skills in 11 categories, with aliases and related skills
│   ├── qualifications.json       80 degree / certification phrases for the EntityRuler
│   ├── custom_stopwords.txt      59 domain stop-words ("candidate", "looking", ...)
│   ├── jd_boilerplate_cues.txt   cue phrases for legal / benefits text in job ads
│   └── background_idf.json       document frequencies from 2,001 real job postings (for IDF)
│
├── scripts/
│   └── build_background_idf.py   rebuilds background_idf.json from public job postings
│
├── sample_data/
│   ├── sample_profile.txt
│   └── sample_job.txt
│
└── tests/                        58 pytest tests (unit + end-to-end)
```

---

## 2. Installation (Windows + VS Code)

Use **Python 3.10, 3.11 or 3.12** (3.11 recommended). Very new versions such as 3.13/3.14 may not have
pre-built wheels for spaCy or PyTorch yet.

Open the project folder in VS Code, then open a terminal (`Ctrl + ` `):

```powershell
# 1. create and activate a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. (optional, recommended) CPU-only PyTorch - a much smaller download than the default
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. install the libraries
pip install -r requirements.txt

# 4. download the pretrained models (needs internet once)
python download_models.py
```

On macOS / Linux, use `python3.11 -m venv .venv` and `source .venv/bin/activate` instead.

In VS Code, select the interpreter `.venv` (*Ctrl+Shift+P → Python: Select Interpreter*).

### Model downloads (what `download_models.py` does)

| Resource | Size | Manual command |
|---|---|---|
| spaCy `en_core_web_sm` | ~13 MB | `python -m spacy download en_core_web_sm` |
| NLTK stop-words | <1 MB | `python -c "import nltk; nltk.download('stopwords')"` |
| Sentence-BERT `all-MiniLM-L6-v2` | ~90 MB | downloaded automatically on first use, then cached in `~/.cache/huggingface` |

After this, the app works **fully offline**.

---

## 3. Running

```powershell
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Click **Load sample data**, then **Analyze**.

Other commands:

```powershell
python -m modules.pipeline   # command-line demo on the sample data (no UI)
python -m pytest             # run all 58 tests
```

---

## 4. Sample input

`sample_data/sample_profile.txt`

> Third-year Artificial Intelligence and Machine Learning engineering student with experience in Python,
> machine learning, NLP, SQL, MongoDB, React, Node.js and Git. Developed projects involving stock
> prediction, game recommendation and text analysis.

`sample_data/sample_job.txt`

> We are looking for an AI/ML intern with knowledge of Python, machine learning, deep learning, NLP, SQL,
> TensorFlow, PyTorch, Git and AWS. Candidates should have experience working with data analysis and
> developing machine learning applications.

**Actual output** (from `python -m modules.pipeline`):

| | |
|---|---|
| Overall match | **53.9 % (Partial match)** = 0.40 × 59.3 + 0.30 × 54.5 + 0.30 × 46.2 |
| Matched skills | Artificial Intelligence, Git, Machine Learning, NLP, Python, SQL |
| Missing skills | **TensorFlow, PyTorch, AWS**, Deep Learning, Data Analysis |
| Profile-only skills | MongoDB, Node.js, React, Text Analytics |
| Top job keywords | nlp, deep learning, tensorflow, machine learning, pytorch, intern, git, data analysis, ml, aws, sql, python |
| Synonym matches | "AI" ↔ "Artificial Intelligence", "ML" ↔ "machine learning" |

---

## 5. The NLP pipeline, step by step

For each technique: **what** it does, **why** it's used, its **input → output**, and **how it feeds the result**.

### Step 0: Text extraction (`text_extraction.py`)
- **What:** reads text from a PDF (`pypdf`) or TXT file (UTF-8, falling back to the Windows cp1252 encoding).
- **Why:** every later step works on plain text. A LinkedIn profile exported with *More → Save to PDF* works directly.
- **In → out:** file bytes → string. A scanned PDF has no text layer, so the user is asked to paste the text instead (OCR is out of scope).

### Step 1: Preprocessing (`preprocessing.py`)
Every intermediate result is kept and shown in the **Pipeline internals** tab.

| # | Step | How | Example |
|---|---|---|---|
| 0 | Boilerplate removal (job only) | drop sentences containing cue phrases (`jd_boilerplate_cues.txt`) | "We are an equal opportunity employer…" → removed |
| 1 | Noise cleaning | regex: URLs, e-mails, phone numbers, "Page 1 of 3", bullets | "• Python" → "Python" |
| 2 | Tokenization | spaCy's rule-based tokenizer | "Node.js, C++" → `Node.js` `,` `C++` |
| 3 | Lowercasing | `token.lower_` | "Python" → "python" |
| 4 | Noise-token removal | drop punctuation, numbers, single characters | `,` `3` removed |
| 5 | Stop-word removal | NLTK English list + 59 domain words | "we", "are", "candidate" removed |
| 6 | Lemmatization | spaCy lemmatizer (+ tiny override list) | "applications" → "application", "developing" → "develop" |

- **Why:** the same concept must always look the same to TF-IDF ("Developing" = "develop").
- **Phrase segments:** words that were adjacent in the original text (with no stop-word or punctuation between them) form a *segment*. Only nouns, adjectives and noun-modifying gerunds can form a phrase, following the part-of-speech filter for technical terms of Justeson & Katz (1995). So "machine learning" is a candidate phrase, while "python, machine" and "develop machine" are not.
- **Two fixes found by testing:** spaCy lemmatizes "data" as "datum", and its `like_url` check treats "node.js" as a website. Both are corrected in the code.
- **Output:** tokens at every stage, final lemmas, sentence list (for embeddings), segments (for TF-IDF) and the spaCy `Doc` (for NER).

### Step 2: Skill extraction (`skill_extraction.py`)
- **What:** finds the skills from `data/skills.json` (140 skills, 11 categories) in both texts and compares them.
- **Why:** a generic NER model does not know that "PyTorch" is a skill (see step 4). A curated dictionary (a *gazetteer*) is precise, explainable and easy to extend.
- **How:** each alias becomes a regular expression with custom word boundaries:
  - "Java" ≠ "JavaScript" and "SQL" ≠ "MySQL", but "C++", "C#" and "Node.js" still match;
  - "machine-learning" = "machine learning";
  - case-sensitive aliases separate "Excel" the tool from "excel" the verb;
  - the longest match wins ("React Native" is not also counted as "React").
- **Gap comparison:** set operations give **matched**, **missing** and **profile-only** skills. **Partial** means the job's exact skill is absent but a more specific related skill is present (job asks for SQL, profile shows MySQL).
- **Employer detection:** a skill used as the employer's name ("At Figma…", "Figma's") is not treated as a requirement.
- **Output → score:** *skill coverage* = (matched + 0.5 × partial) / skills requested by the job.
- **Extending:** add `{"name": "Rust", "aliases": ["rust"]}` to a category in `skills.json`. No code change is needed.

### Step 3: TF-IDF keyword analysis (`tfidf_analysis.py`)
- **What:** weights every term (unigram or established bigram) by TF-IDF, ranks the job's most important keywords, and checks which ones appear in the profile.
- **Why:** ATS (applicant-tracking) systems and recruiters look for the job's key terms. TF-IDF is the standard way to find the terms that characterise a document (Salton & Buckley, 1988; Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, 2008, ch. 6).
- **Formula** (the same as scikit-learn's `TfidfVectorizer(sublinear_tf=True, smooth_idf=True)`; a test proves the numbers are identical):

  ```
  tf(t,d) = 1 + ln(count of t in d)
  idf(t)  = ln((1 + N) / (1 + df(t))) + 1
  w(t,d)  = tf × idf, and each vector is scaled to length 1 (L2 norm)
  ```
- **Where IDF comes from:** IDF needs many documents. One job description cannot tell that "team" is generic and "pytorch" is specific. So document frequencies come from a **reference corpus of 2,001 real job postings from 35 companies** (Greenhouse public job-board API, every role type, built 2026-10-06 by `scripts/build_background_idf.py`). Only counts are stored, no posting text. Measured IDF values:

  | term | df (of 2,001) | idf |
  |---|---|---|
  | team | 1,968 | 1.02 |
  | customer | 1,304 | 1.43 |
  | python | 462 | 2.46 |
  | machine learning | 234 | 3.14 |
  | pytorch | 29 | 5.20 |
  | nlp | 16 | 5.77 |
- **Phrase rule:** a bigram can only become a keyword if it is a known skill phrase or appears in ≥ 5 corpus postings. Otherwise accidental word pairs ("diverse enterprise") get the maximum IDF and crowd out real keywords, which is a known weakness of IDF on sparse n-grams that real postings exposed. A unigram that is part of a chosen phrase is not listed again.
- **Synonyms:** a job keyword also counts as present if the profile uses a skill alias for it ("AI" ↔ "Artificial Intelligence").
- **Output → score:** *keyword coverage* = Σ weight(found keywords) / Σ weight(top keywords). The TF-IDF cosine of the two documents is also shown for reference.

### Step 4: Named Entity Recognition (`ner_analysis.py`)
- **What:** spaCy `en_core_web_sm` finds ORG, GPE (places), DATE, MONEY, PERSON and other entities. An **EntityRuler** placed *before* the statistical NER adds a custom **QUALIFICATION** label (B.Tech, PhD, certifications…).
- **Why:** entities add structured context (employers, locations, required degrees), and qualifications the job mentions feed a recommendation.
- **Limitation, demonstrated on purpose:** the statistical model was trained on OntoNotes 5 (news and web text), not on resumes. Observed on the sample data: "Python" → GPE (a place), "Git" → PERSON, "TensorFlow"/"PyTorch"/"Node.js" → ORG. The **Entities** tab shows this side by side with the dictionary results. That is the justification for using a dictionary for skill extraction.

### Step 5: Semantic embeddings and cosine similarity (`semantic_similarity.py`)
- **What:** each sentence becomes a 384-dimensional vector from the pretrained **Sentence-BERT** model `all-MiniLM-L6-v2` (Reimers & Gurevych, 2019). The document vector is the mean of its sentence vectors.
- **Why:** TF-IDF only sees exact words. "Trained CNNs to classify images" and "deep learning for computer vision" share no keywords but mean almost the same thing.
- **Cosine similarity:** `cos(a,b) = a·b / (‖a‖‖b‖)`, written out explicitly in `cosine_similarity()`. Percentage = max(0, cos) × 100.
- **Sentence level:** for every job sentence, the best-matching profile sentence is shown. Requirements with low similarity become recommendations.
- **Why mean-of-sentences:** MiniLM reads at most 256 word-pieces, so embedding a long profile in one piece would cut it off.

### Step 6: Scoring. See section 6.

### Step 7: Gap analysis → strengths and recommendations (`recommendations.py`)
Transparent IF-THEN rules over the results (no generative AI), so every suggestion traces back to a detected gap:
- missing technical skills (High), grouped by category;
- professional skills such as communication (Medium), with advice to show real examples;
- partial skills: use the job's exact term, if it is accurate;
- missing important keywords that aren't already covered by a missing skill;
- weakly addressed job requirements. A sentence counts as a requirement only if it names a technical skill, or contains a top keyword and is not written in the company's voice ("we", "our");
- qualifications the job mentions, profile length, absence of measurable results, and skill ordering.

**Honesty rule:** recommendations never say "add X". They say: *if you have genuinely used X, mention it with what you built; if not, build it first. Do not list skills you have not used.*

---

## 6. Scoring methodology (`scoring.py`)

```
Overall = 0.40 × Semantic similarity + 0.30 × Skill coverage + 0.30 × Keyword coverage
```

| Component | Weight | Rationale |
|---|---|---|
| Semantic similarity | 40 % | broadest view: captures paraphrases and overall relevance |
| Skill coverage | 30 % | most concrete evidence, and what recruiters filter on |
| Keyword coverage (TF-IDF) | 30 % | ATS-style overlap; catches domain and role terms outside the skill list |

Why these three together: each single metric fails in a predictable way.
- Semantic similarity rewards text that *sounds* similar even when the required tools are missing.
- Skill coverage ignores everything that isn't in the dictionary.
- Keyword matching misses synonyms and paraphrases.

Combining them is more robust, and every point can still be traced to a component. The dashboard prints the exact sum, e.g. `0.40 × 59.3 + 0.30 × 54.5 + 0.30 × 46.2 = 53.9`.

- If a component cannot be computed (e.g. the job names no dictionary skill), its weight is **redistributed proportionally** instead of counting it as 0 %.
- Bands: ≥ 75 Strong · ≥ 55 Good · ≥ 35 Partial · < 35 Weak.
- **This is an analytical text-compatibility score, not a probability of being shortlisted or hired.** It cannot see experience that is not written in the text.

### Behaviour on real data (validation run)

The developer's own LinkedIn PDF export and CV were scored against four real public postings. The Account Executive role is an unrelated control:

| Job posting | CV overall | CV semantic | LinkedIn-PDF overall |
|---|---|---|---|
| Machine Learning Engineer, Platform | **47.4** | 72.6 | 34.9 |
| Applied AI Architect | 45.5 | 62.3 | 38.6 |
| Account Executive (sales, control) | 35.2 | 58.1 | 29.2 |
| Data Science Intern | 33.3 | 62.8 | 32.6 |

The ranking is sensible: the ML roles score highest and the sales role lowest. Two honest observations:
1. Raw MiniLM cosine is **compressed**. Unrelated professional texts still reach about 0.55–0.60, so compare semantic values relative to each other, not as absolute percentages.
2. The CV (557 words) scores higher than the shorter, web-development-focused LinkedIn export (325 words). Richer text gives the analysis more evidence.

---

## 7. How to demonstrate the project to your professor (about 10 minutes)

1. **Problem (1 min):** "Given a profile and a job description, measure compatibility and explain the gaps, using genuine NLP rather than an LLM API."
2. **Run the sample (2 min):** *Load sample data → Analyze*. Point at the score formula under the overall score, then the missing skills TensorFlow / PyTorch / AWS.
3. **Pipeline internals tab (2 min):** open the preprocessing expanders and walk through cleaning → tokens → stop-words removed → lemmas ("developing" → "develop", "applications" → "application"). Show the boilerplate sentences removed from a real job ad.
4. **Keywords tab (2 min):** explain the IDF column. "team" has IDF ≈ 1 because almost every posting in the 2,001-posting corpus contains it, while "pytorch" has ≈ 5.2. Point out the synonym match "AI" ↔ "Artificial Intelligence".
5. **Entities tab (1 min):** show that spaCy labels Python as a place and Git as a person, and explain why skills use a dictionary instead.
6. **Semantic tab (1 min):** the job-sentence ↔ best-profile-sentence table, and why embeddings catch paraphrases that TF-IDF misses.
7. **Live experiment (1 min):** add *"Built an image classifier with PyTorch and deployed it on AWS"* to the profile (it's a demo; say so) and re-analyze. PyTorch and AWS move to *Matched*, Deep Learning becomes *Partial* (PyTorch is related evidence) and the score rises from 53.9 to 66.4.
8. **Code (optional):** `modules/pipeline.py` shows the whole pipeline in about 60 lines; `python -m pytest` runs 58 tests, including one proving the TF-IDF matches scikit-learn exactly.

Likely questions:
- *Why not just use ChatGPT?* It isn't explainable or reproducible, and it isn't the NLP being assessed.
- *Why these weights?* See section 6.
- *How do you know the TF-IDF is right?* `test_manual_tfidf_equals_scikit_learn`.
- *Limitations?* See below.

### Limitations (worth stating up front)
- The skill dictionary only knows the skills it contains (140); an unlisted skill is invisible to skill coverage, though TF-IDF and embeddings still see it.
- The reference corpus comes mostly from US/European tech companies, which may weight some Indian job-ad terms slightly differently.
- Boilerplate detection uses cue phrases, so unusual legal wording can slip through.
- Scanned PDFs need OCR, which isn't included.
- English only.

---

## 8. Common errors and fixes

| Error | Fix |
|---|---|
| `OSError: [E050] Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` (inside the activated venv). |
| `spacy download` times out (GitHub blocked by a firewall or proxy) | Install from the Hugging Face mirror: `pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl` (this mirror is v3.7.1; it works with spaCy 3.8 but may print a version warning). |
| `LookupError: Resource stopwords not found` | `python -c "import nltk; nltk.download('stopwords')"`. The app also falls back to spaCy's stop-word list automatically. |
| "Could not load the sentence-embedding model" | The first run needs internet (~90 MB). Run `python download_models.py` on a working network; after that it is cached. |
| `'streamlit' is not recognized…` | The venv is not active: run `.venv\Scripts\activate`, or use `python -m streamlit run app.py`. |
| PowerShell: "running scripts is disabled on this system" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again (or use `cmd`, where the script is `.venv\Scripts\activate.bat`). |
| `pip` fails building spaCy / no wheel found | You are on a too-new Python. Install Python 3.11 and recreate the venv. |
| `OSError: [WinError 126] … c10.dll` / torch DLL load failed | Install the Microsoft Visual C++ Redistributable (x64), then restart the terminal. |
| `ModuleNotFoundError: No module named 'modules'` | Run commands from the project root folder (where `app.py` is). |
| "No selectable text found in this PDF" | The PDF is a scanned image. Paste the text instead. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| First analysis takes 10–20 s | Normal: the models load once and are cached; later runs take well under a second. |

---

## 9. Rebuilding the reference corpus (optional)

```powershell
python scripts/build_background_idf.py
```

This fetches the **current** public postings (needs internet, takes a few minutes) and rewrites
`data/background_idf.json`. The shipped file keeps results reproducible offline. A rebuild will change the
counts slightly because job boards change daily.

## 10. Privacy

Profile text is processed in memory on your computer. It is never stored, logged or sent anywhere, and
Streamlit's usage statistics are disabled in `.streamlit/config.toml`.
