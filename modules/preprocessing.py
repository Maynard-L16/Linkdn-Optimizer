"""Step 1 - Text preprocessing.

WHAT : Turns raw profile / job text into clean, normalised tokens.
WHY  : Raw text is noisy (URLs, page numbers, punctuation, "the", "and",
       "developing" vs "developed"). TF-IDF and keyword matching only work
       well when the same concept is always written the same way.
INPUT: a raw string.
OUTPUT: PreprocessResult holding the output of EVERY step (so the UI can show
       the professor each transformation), plus sentence lists for the
       semantic module and the spaCy Doc for the NER module.

Steps, in order:
    0. (job descriptions only) Boilerplate removal - sentences that are legal /
                             benefits / privacy text (cue phrases in
                             data/jd_boilerplate_cues.txt), not requirements
    1. Noise cleaning      - regex removal of URLs, e-mails, phone numbers,
                             "Page x of y", bullet symbols, extra whitespace
    2. Tokenization        - spaCy's rule-based tokenizer (handles "Node.js",
                             "C++", "don't" better than str.split)
    3. Lowercasing         - "Python" and "python" become the same token
    4. Noise-token removal - drop punctuation, numbers, single characters
    5. Stop-word removal   - NLTK English list + domain words (data/custom_stopwords.txt)
    6. Lemmatization       - spaCy's lemmatizer: "applications" -> "application",
                             "developing" -> "develop"
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from spacy.tokens import Doc, Span, Token

from modules.nlp_resources import BOILERPLATE_CUES_PATH, get_nlp, get_stopwords, read_word_list

# --- Step 1 regexes ---------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+|linkedin\.com/\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s\-()]{8,}\d(?!\w)")
PAGE_RE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
BULLET_RE = re.compile(r"[•‣▪●◦⁃∙➢►]")
# A line break followed by a lowercase letter is almost always a PDF line-wrap
# in the middle of a sentence, so we join it back with a space.
WRAPPED_LINE_RE = re.compile(r"\n(?=[a-z])")

# A "sentence" for boilerplate detection: text up to . ! ? or a line break.
SENTENCE_CHUNK_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)")

# --- Step 4: a "meaningful" token starts with a letter and may contain
# letters, digits and the symbols used in tech names (c++, c#, node.js).
VALID_TOKEN_RE = re.compile(r"^[a-z][a-z0-9+#.\-]*$")

# spaCy's lemmatizer follows strict English grammar ("data" -> "datum"),
# which is wrong for how the word is used in tech writing.
# Only nouns, proper nouns and adjectives may form multi-word phrases. A verb
# ("develop") becomes a single-word segment, so the bigram "develop machine"
# is never created. This is the part-of-speech filter for technical terms of
# Justeson & Katz (1995): terms are (ADJ|NOUN)* NOUN sequences.
PHRASE_POS = {"NOUN", "PROPN", "ADJ", "X"}

LEMMA_OVERRIDES = {"data": "data", "datum": "data", "analytics": "analytics", "aws": "aws", "kubernetes": "kubernetes"}


@dataclass
class PreprocessResult:
    original: str
    cleaned: str
    lowercased: str
    tokens: list[str]  # step 2+3: every token, lowercased
    tokens_no_noise: list[str]  # step 4
    tokens_no_stopwords: list[str]  # step 5
    lemmas: list[str]  # step 6 (final tokens)
    sentences: list[str]  # cleaned sentence strings (for embeddings)
    # Lemmas grouped as sentence -> segment -> words. A "segment" is a run of
    # words that were adjacent in the original text (no stop-word or
    # punctuation between them). TF-IDF bigrams are only built inside a
    # segment, so "python, machine learning" never yields the fake bigram
    # "python machine".
    sentence_segments: list[list[list[str]]]
    # Lemmas only ever used as verbs/adverbs in this text ("develop", "involve").
    # They stay in the TF-IDF vectors but are not offered as "important keywords".
    non_phrase_lemmas: set[str]
    doc: Doc = field(repr=False)
    removed_boilerplate: list[str] = field(default_factory=list)

    @property
    def segments(self) -> list[list[str]]:
        return [seg for sent in self.sentence_segments for seg in sent]

    @property
    def processed_text(self) -> str:
        return " ".join(self.lemmas)


def clean_text(text: str) -> str:
    """Step 1: remove noise that carries no meaning but keep sentence structure."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = PAGE_RE.sub(" ", text)
    text = BULLET_RE.sub("\n", text)
    text = WRAPPED_LINE_RE.sub(" ", text)
    text = re.sub(r"[ \t ]+", " ", text)  # collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)  # collapse blank lines
    return text.strip()


def remove_boilerplate(text: str) -> tuple[str, list[str]]:
    """Step 0: drop sentences containing a boilerplate cue phrase.

    Returns (text without those sentences, list of removed sentences).
    """
    cues = read_word_list(BOILERPLATE_CUES_PATH)
    removed: list[str] = []

    def drop(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if any(cue in chunk.lower() for cue in cues):
            removed.append(" ".join(chunk.split()))
            return ""
        return chunk

    return SENTENCE_CHUNK_RE.sub(drop, text), removed


def _sentence_spans(doc: Doc) -> list[Span]:
    """Sentences from spaCy's parser, additionally split at line breaks.

    Resumes and LinkedIn profiles are full of bullet points and headings that
    have no full stop; each line is treated as its own "sentence".
    """
    spans: list[Span] = []
    for sent in doc.sents:
        start = sent.start
        for tok in sent:
            if tok.is_space and "\n" in tok.text:
                if tok.i > start:
                    spans.append(doc[start : tok.i])
                start = tok.i + 1
        if start < sent.end:
            spans.append(doc[start : sent.end])
    return [s for s in spans if s.text.strip()]


def _is_intra_word_hyphen(tok: Token) -> bool:
    """True for the '-' in 'third-year' (no spaces around it)."""
    return tok.text == "-" and not tok.whitespace_ and tok.i > 0 and not tok.doc[tok.i - 1].whitespace_


def preprocess(text: str, drop_boilerplate: bool = False) -> PreprocessResult:
    """Run all preprocessing steps and keep every intermediate result."""
    stopwords, _ = get_stopwords()
    removed: list[str] = []
    cleaned = clean_text(text)
    if drop_boilerplate:
        cleaned, removed = remove_boilerplate(cleaned)
        cleaned = clean_text(cleaned)
    doc = get_nlp()(cleaned)  # tokenization + tagging + lemmatization + NER in one pass

    tokens: list[str] = []
    no_noise: list[str] = []
    no_stop: list[str] = []
    lemmas: list[str] = []
    sentences: list[str] = []
    sentence_segments: list[list[list[str]]] = []
    phrase_lemmas: set[str] = set()
    other_lemmas: set[str] = set()

    for span in _sentence_spans(doc):
        sentences.append(span.text.strip())
        segments: list[list[str]] = []
        current: list[str] = []

        def close_segment() -> None:
            nonlocal current
            if current:
                segments.append(current)
            current = []

        for tok in span:
            if tok.is_space:
                continue
            lower = tok.lower_  # steps 2 + 3
            tokens.append(lower)

            if _is_intra_word_hyphen(tok):
                continue  # keep "third-year" together as one segment
            # (URLs/e-mails were already removed in step 1; spaCy's like_url would
            # wrongly flag tech names such as "node.js".)
            if tok.is_punct or not VALID_TOKEN_RE.match(lower) or len(lower) < 2:
                close_segment()  # step 4: noise breaks the phrase
                continue
            no_noise.append(lower)

            lemma = LEMMA_OVERRIDES.get(lower) or tok.lemma_.lower().strip()
            if lower in stopwords or lemma in stopwords:
                close_segment()  # step 5: stop-word breaks the phrase
                continue
            no_stop.append(lower)

            lemma = lemma if VALID_TOKEN_RE.match(lemma or "") else lower  # step 6
            # A gerund directly after a noun is part of a compound term, not a
            # verb: spaCy tags "learning" in "machine learning projects" as
            # VERB/VBG. Such a gerund may continue a phrase but never start one,
            # so "developing applications" is still split.
            continues_phrase = tok.tag_ == "VBG" and bool(current)
            if continues_phrase:
                lemma = lower  # keep the noun form: "machine learning", not "machine learn"
            lemmas.append(lemma)
            if tok.pos_ in PHRASE_POS or continues_phrase:
                phrase_lemmas.add(lemma)
                current.append(lemma)
            else:  # verb/adverb/etc.: a one-word segment of its own
                other_lemmas.add(lemma)
                close_segment()
                segments.append([lemma])
        close_segment()
        sentence_segments.append(segments)

    return PreprocessResult(
        original=text,
        cleaned=cleaned,
        lowercased=cleaned.lower(),
        tokens=tokens,
        tokens_no_noise=no_noise,
        tokens_no_stopwords=no_stop,
        lemmas=lemmas,
        sentences=sentences,
        sentence_segments=sentence_segments,
        non_phrase_lemmas=other_lemmas - phrase_lemmas,
        doc=doc,
        removed_boilerplate=removed,
    )
