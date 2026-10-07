"""Build data/background_idf.json - the reference corpus statistics used for IDF.

WHY: IDF ("inverse document frequency") needs MANY documents to tell common
words ("team", "time") from specific ones ("pytorch"). One job description
cannot provide that, so we compute document frequencies once over real public
job postings and ship only those counts (no posting text is stored).

SOURCE: Greenhouse public Job Board API - https://developers.greenhouse.io/job-board.html
        (public, no authentication). Every role type at each company is
        included (engineering, sales, operations, finance, ...), so the corpus
        represents job postings in general, not only tech roles.

BIAS NOTE: the boards are mostly US/European technology companies, so terms
common in, for example, Indian campus-hiring postings may be weighted
slightly differently.

Run (needs internet, takes a few minutes):
    python scripts/build_background_idf.py
The shipped data/background_idf.json makes the app reproducible offline;
re-running fetches the CURRENT postings, so counts will change slightly.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.nlp_resources import DATA_DIR  # noqa: E402
from modules.preprocessing import preprocess  # noqa: E402
from modules.tfidf_analysis import segment_ngrams, to_document  # noqa: E402

BOARDS = [
    "airbnb", "stripe", "dropbox", "pinterest", "reddit", "discord", "duolingo", "gitlab", "coinbase",
    "robinhood", "lyft", "instacart", "asana", "cloudflare", "twilio", "mongodb", "elastic", "datadog",
    "brex", "gusto", "squarespace", "peloton", "okta", "samsara", "affirm", "chime", "hellofresh", "toast",
    "flexport", "nextdoor", "monzo", "databricks", "scaleai", "anthropic", "figma",
]
MAX_PER_BOARD = 60  # cap so no single company's wording dominates
MIN_DF = 2  # terms seen in only one posting are treated as unseen (maximum IDF)
SEED = 42
OUTPUT = DATA_DIR / "background_idf.json"


def fetch_board(board: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp).get("jobs", [])


def html_to_text(content: str) -> str:
    raw = html.unescape(content or "")
    raw = re.sub(r"</(p|li|h\d|div)>|<br\s*/?>", "\n", raw)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def main() -> None:
    rng = random.Random(SEED)
    df: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    used_boards: list[str] = []
    n_docs = 0
    for board in BOARDS:
        try:
            jobs = fetch_board(board)
        except Exception as exc:  # a board can disappear; skip it
            print(f"  skip {board}: {exc}")
            continue
        jobs = sorted(jobs, key=lambda j: j["id"])
        rng.shuffle(jobs)
        taken = 0
        for job in jobs:
            if taken >= MAX_PER_BOARD:
                break
            text = html_to_text(job.get("content", ""))
            digest = hashlib.sha1(" ".join(text.split()).encode()).hexdigest()
            if len(text.split()) < 100 or digest in seen_hashes:  # skip stubs and duplicate postings
                continue
            seen_hashes.add(digest)
            result = preprocess(text, drop_boilerplate=True)
            df.update(set(segment_ngrams(to_document(result.segments))))  # each term once per document
            taken += 1
        if taken:
            used_boards.append(board)
            n_docs += taken
            print(f"  {board:12s} {taken:3d} postings (total {n_docs})")

    kept = {term: count for term, count in sorted(df.items()) if count >= MIN_DF}
    OUTPUT.write_text(
        json.dumps(
            {
                "meta": {
                    "source": "Greenhouse public Job Board API (boards-api.greenhouse.io)",
                    "boards": used_boards,
                    "built_on": date.today().isoformat(),
                    "max_postings_per_board": MAX_PER_BOARD,
                    "min_df": MIN_DF,
                    "note": "Document frequencies only; no posting text is stored.",
                },
                "n_documents": n_docs,
                "df": kept,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} - {n_docs} documents, {len(kept):,} terms (df >= {MIN_DF}).")


if __name__ == "__main__":
    main()
