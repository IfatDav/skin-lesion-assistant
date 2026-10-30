"""
rag.py · The grounded guidance assistant over NCI PDQ (Module 5 method, capstone corpus)
=======================================================================================

Charter: "Retrieval asset: the module 5 grounded assistant", M3: groundedness >= 0.9 on 20 frozen
guidance questions (5 unanswerable, 5/5 refusals), plus a check that each number appears in its
cited source.

Same method as the Module 5 assignment, with the NCI PDQ summaries as the corpus:
  corpus      7 NCI PDQ Health Professional summaries on skin cancer, saved once (nci_corpus.json)
              so every run uses the same data
  chunking    by section, ~900 characters with ~150 overlap (PDQ sections are long, unlike abstracts)
  citation id every chunk has an id such as  pdq:screening:s04:c02  (the role arXiv ids played)
  embedding   sentence-transformers/all-MiniLM-L6-v2, normalized, in Chroma
  answer()    answers ONLY from retrieved chunks, cites their ids, or says exactly "I don't know"
  metrics     groundedness (cited ids that were retrieved), refusal, numbers found in cited chunks
  improvement rerank=True: retrieve wide (k=20) and re-rank narrow with a cross-encoder

The agent's tool `retrieve_guidance_context` calls retrieve() and cites the same ids.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. Corpus
# ──────────────────────────────────────────────────────────────────────────────
# Health Professional versions: the reader is a dermatologist (feedback: "what does a
# dermatologist learn from a summary written for patients?").
PDQ_DOCS = {
    "screening":      ("Skin Cancer Screening (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/skin-screening-pdq"),
    "prevention":     ("Skin Cancer Prevention (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/skin-prevention-pdq"),
    "melanoma_tx":    ("Melanoma Treatment (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/melanoma-treatment-pdq"),
    "skin_tx":        ("Skin Cancer Treatment (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/skin-treatment-pdq"),
    "genetics":       ("Genetics of Skin Cancer (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/skin-genetics-pdq"),
    "child_melanoma": ("Childhood Melanoma Treatment (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/child-melanoma-treatment-pdq"),
    "child_skin":     ("Childhood BCC and SCC of the Skin Treatment (PDQ) - HP", "https://www.cancer.gov/types/skin/hp/child-skin-treatment-pdq"),
}
SKIP_SECTIONS = ("latest updates to this summary", "current clinical trials", "references")
STOP_SECTION = "about this pdq summary"

BASE = Path(os.getenv("RAG_DIR", "."))
CORPUS_JSON = BASE / "nci_corpus.json"     # saved corpus: commit it to git
CHROMA_DIR = BASE / "chroma_nci"           # rebuilt from the corpus: not in git
COLLECTION = "nci_pdq_skin"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_CHARS = 900
OVERLAP_CHARS = 150
CITE_RE = re.compile(r"pdq:[a-z_]+:s\d{2}:c\d{2}")
REFUSAL = "I don't know"


def _clean(text: str) -> str:
    text = re.sub(r"\[\s*PUBMED Abstract\s*\]", "", text, flags=re.I)
    text = re.sub(r"\[\d+(?:[,–-]\d+)*\]", "", text)              # citation markers [3], [4-6]
    text = re.sub(r"\[Level of evidence[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:)])", r"\1", text)


def parse_pdq_html(html: str) -> list[dict]:
    """One PDQ page -> [{section, text}] in reading order, boilerplate dropped."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in root.find_all(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    sections, current, buf, skipping = [], "Introduction", [], False

    def flush():
        text = _clean(" ".join(buf))
        if text and not skipping:
            sections.append({"section": current, "text": text})
        buf.clear()

    for el in root.find_all(["h2", "h3", "h4", "p", "li"]):
        if el.name in ("h2", "h3", "h4"):
            title = _clean(el.get_text(" "))
            if not title:
                continue
            flush()
            low = title.lower()
            if low.startswith(STOP_SECTION):
                break
            skipping = any(low.startswith(s) for s in SKIP_SECTIONS)
            current = title
            continue
        if el.find_parent("li") is not None:
            continue
        text = el.get_text(" ")
        if "PUBMED Abstract" in text or len(text.strip()) < 25:
            continue
        buf.append(text)
    flush()
    return sections


def fetch_corpus(local_dir: str | None = None) -> dict:
    """Download the PDQ pages. Plan B: save a page from the browser as HTML into local_dir,
    named <key>.html (for example screening.html)."""
    import requests

    corpus = {}
    for key, (title, url) in PDQ_DOCS.items():
        local = Path(local_dir) / f"{key}.html" if local_dir else None
        try:
            if local and local.exists():
                html = local.read_text(encoding="utf-8", errors="ignore")
            else:
                r = requests.get(url, timeout=30, headers={"User-Agent": "DS23-course-project"})
                r.raise_for_status()
                html = r.text
        except Exception as e:
            print(f"  ! {key}: {e}")
            continue
        corpus[key] = {"title": title, "url": url, "sections": parse_pdq_html(html)}
        print(f"  {key:15s} {len(corpus[key]['sections']):3d} sections")
    return corpus


def load_or_fetch_corpus(refresh: bool = False, local_dir: str | None = None) -> dict:
    if CORPUS_JSON.exists() and not refresh:
        return json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    corpus = fetch_corpus(local_dir)
    if not corpus:
        raise RuntimeError("no PDQ page could be loaded (see Plan B in fetch_corpus)")
    BASE.mkdir(parents=True, exist_ok=True)
    CORPUS_JSON.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    return corpus


# ──────────────────────────────────────────────────────────────────────────────
# 2. Chunking (justification in the decisions report: PDQ sections run to thousands of
#    characters, so whole-section chunks would dilute retrieval; 900/150 keeps ~one idea per chunk)
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s])) > size:
            chunks.append(" ".join(cur))
            tail, n = [], 0
            for prev in reversed(cur):
                if n + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                n += len(prev) + 1
            cur = tail
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_chunks(corpus: dict) -> list[dict]:
    rows = []
    for key, d in corpus.items():
        for si, sec in enumerate(d["sections"]):
            for ci, ch in enumerate(chunk_text(sec["text"])):
                rows.append({"id": f"pdq:{key}:s{si:02d}:c{ci:02d}", "text": ch,
                             "meta": {"doc": key, "source": d["title"], "section": sec["section"], "url": d["url"]}})
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 3. Index
# ──────────────────────────────────────────────────────────────────────────────
_EMBEDDER = _RERANKER = _COL = None
_CHUNK_TEXT: dict[str, str] | None = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def _collection(reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION)


def build_index(refresh_corpus: bool = False, local_dir: str | None = None) -> int:
    global _COL, _CHUNK_TEXT
    corpus = load_or_fetch_corpus(refresh_corpus, local_dir)
    rows = make_chunks(corpus)
    emb = _embedder().encode([r["text"] for r in rows], normalize_embeddings=True, show_progress_bar=True)
    col = _collection(reset=True)
    for i in range(0, len(rows), 200):
        b = rows[i:i + 200]
        col.add(ids=[r["id"] for r in b], documents=[r["text"] for r in b],
                metadatas=[r["meta"] for r in b], embeddings=[e.tolist() for e in emb[i:i + 200]])
    _COL, _CHUNK_TEXT = col, None
    print(f"indexed {len(rows)} chunks from {len(corpus)} documents")
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Retrieve (the agent's tool calls this)
# ──────────────────────────────────────────────────────────────────────────────
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0")) or None   # frozen in the RAG notebook


def retrieve(query: str, k: int = 3, max_distance: float | None = MAX_DISTANCE,
             rerank: bool = False, wide_k: int = 20, where: dict | None = None) -> list[dict]:
    """Up to k chunks: {id, quote, source, section, url, distance}. [] = nothing relevant."""
    global _COL, _RERANKER
    if _COL is None:
        _COL = _collection()
    q = _embedder().encode([query], normalize_embeddings=True)[0].tolist()
    res = _COL.query(query_embeddings=[q], n_results=wide_k if rerank else k, where=where)
    hits = [{"id": i, "quote": d, "source": m["source"], "section": m["section"], "url": m["url"],
             "distance": round(float(dist), 4)}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
    if max_distance is not None:
        hits = [h for h in hits if h["distance"] <= max_distance]
    if rerank and hits:
        if _RERANKER is None:
            from sentence_transformers import CrossEncoder
            _RERANKER = CrossEncoder(RERANK_MODEL)
        scores = _RERANKER.predict([(query, h["quote"]) for h in hits])
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda t: -t[0])]
    return hits[:k]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Grounded answer (Module 5 section 6)
# ──────────────────────────────────────────────────────────────────────────────
GROUNDED_SYSTEM = (
    "You answer questions for a dermatologist using ONLY the numbered passages from NCI PDQ "
    "summaries given in CONTEXT. Every sentence of your answer must be supported by a passage, and "
    "you must cite the passage id(s) you used in square brackets, for example [pdq:screening:s04:c02]. "
    "Copy numbers exactly as they appear in the passage. Do not use any outside knowledge. "
    f"If the passages do not contain the answer, reply exactly: {REFUSAL}"
)


def answer(question: str, llm, k: int = 3, **retrieve_kw) -> tuple[str, list[str]]:
    """llm(messages) -> str. Returns (answer, retrieved ids)."""
    hits = retrieve(question, k=k, **retrieve_kw)
    if not hits:
        return REFUSAL, []
    context = "\n\n".join(f"[{h['id']}] ({h['source']} · {h['section']}) {h['quote']}" for h in hits)
    out = llm([{"role": "system", "content": GROUNDED_SYSTEM},
               {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}])
    return out, [h["id"] for h in hits]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Metrics (Module 5 section 4, plus the Charter's number check)
# ──────────────────────────────────────────────────────────────────────────────
def groundedness(answer_text: str, retrieved_ids: list[str]) -> float:
    """Fraction of cited chunk ids that were actually retrieved. No citation -> 0."""
    cited = set(CITE_RE.findall(answer_text))
    if not cited:
        return 0.0
    return len(cited & set(retrieved_ids)) / len(cited)


def is_refusal(text: str) -> bool:
    low = text.lower()
    return "i don't know" in low or "i do not know" in low


def chunk_text_by_id(cid: str) -> str:
    global _CHUNK_TEXT
    if _CHUNK_TEXT is None:
        _CHUNK_TEXT = {r["id"]: r["text"] for r in make_chunks(load_or_fetch_corpus())}
    return _CHUNK_TEXT.get(cid, "")


def numbers_supported(answer_text: str) -> bool:
    """Charter M3: every number in the answer appears in a chunk it cites."""
    body = CITE_RE.sub("", answer_text)
    nums = set(re.findall(r"\d+(?:\.\d+)?", body))
    cited = " ".join(chunk_text_by_id(c) for c in set(CITE_RE.findall(answer_text)))
    return all(n in cited for n in nums)


def is_verbatim(quote: str, source: str | None = None) -> bool:
    """True if `quote` appears word for word in the corpus (in `source`, if given)."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    corpus = load_or_fetch_corpus()
    q = norm(quote)
    if not q:
        return False
    for d in corpus.values():
        if source and d["title"] != source:
            continue
        if q in norm(" ".join(s["text"] for s in d["sections"])):
            return True
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build_index(refresh_corpus="--refresh" in sys.argv)
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        for h in retrieve(" ".join(sys.argv[2:]), max_distance=None):
            print(f"\n[{h['id']}] {h['distance']} · {h['source']} · {h['section']}\n{h['quote'][:400]}")
    else:
        print(__doc__)
