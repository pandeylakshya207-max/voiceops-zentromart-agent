"""Knowledge base: section-level chunking of kb/*.md and hybrid (BM25 + embedding) retrieval
with document-precedence re-ranking (KB-00.1)."""
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import CACHE_DIR, EMBED_MODEL, KB_DIR, TOP_K

# KB-00.1: policy > agent guide > help > FAQ > marketing
PRECEDENCE = {"policy": 0, "meta": 0, "agent_guide": 1, "help": 2, "faq": 3, "marketing": 4}
PREFIX_TYPE = {"pol": "policy", "agent": "agent_guide", "help": "help", "faq": "faq", "mkt": "marketing", "00": "meta"}

SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.M)
TOKEN_RE = re.compile(r"[a-z0-9ऀ-ॿ]+")

# Everyday / spoken / Hindi words -> the vocabulary the policies use
SYNONYMS = {
    "damaged": ["torn", "broken", "cracked", "dented", "scratched", "leaking", "leaked", "spoiled", "smashed",
                "defective", "faulty", "ripped", "crushed", "tuta", "toota", "kharab", "phata", "fata"],
    "wrong item": ["different item", "galat", "wrong product", "not what i ordered"],
    "missing": ["not in the box", "empty box", "gayab", "nahi mila"],
    "refund": ["money back", "paisa wapas", "paise wapas", "paisa", "paise", "\u0930\u093f\u092b\u0902\u0921", "\u092a\u0948\u0938\u0947"],
    "return": ["send back", "wapas", "lautana", "\u0932\u094c\u091f\u093e\u0928\u0947", "\u0935\u093e\u092a\u0938", "\u0930\u093f\u091f\u0930\u094d\u0928"],
    "cancel": ["cancellation", "cancel karna", "\u0930\u0926\u094d\u0926", "\u0915\u0948\u0902\u0938\u093f\u0932"],
    "delivery": ["arrive", "aayega", "pahunch", "\u0921\u093f\u0932\u0940\u0935\u0930\u0940", "kab milega", "shipping"],
    "warranty": ["guarantee", "repair", "\u0935\u093e\u0930\u0902\u091f\u0940"],
    "reverse pickup fee": ["pickup fee", "pickup charge", "return fee"],
    "cash on delivery": ["cod", "cash"],
}


def expand_query(q: str) -> str:
    low = q.lower()
    extra = [canon for canon, alts in SYNONYMS.items() if any(a in low for a in alts)]
    return q + (" " + " ".join(extra) if extra else "")


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_file: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    section_title: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return PRECEDENCE.get(self.doc_type, 3)

    def header(self) -> str:
        sup = " [SUPERSEDED]" if self.status == "superseded" else ""
        return (f"[{self.id}] {self.doc_title} — {self.section_title} "
                f"(type: {self.doc_type}, status: {self.status}{sup}, effective {self.effective_from})")

    def render(self, max_chars: int = 1400) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " …"
        return f"{self.header()}\n{body}"


def _frontmatter(raw: str):
    meta, body = {}, raw
    m = re.match(r"^\s*---\s*\n(.*?)\n---\s*\n", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"\'')
        body = raw[m.end():]
    return meta, body


def load_chunks() -> list:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _frontmatter(raw)
        prefix = path.stem.split("_")[0]
        doc_type = meta.get("doc_type") or PREFIX_TYPE.get(prefix, "help")
        title_m = re.search(r"^#\s+(.+)$", body, re.M)
        doc_title = meta.get("title") or (title_m.group(1).strip() if title_m else path.stem)
        status = meta.get("status", "current")
        if "superseded" in path.stem:
            status = "superseded"
        common = dict(doc_id=meta.get("doc_id", path.stem), doc_file=path.name, doc_title=doc_title,
                      doc_type=doc_type, status=status, effective_from=meta.get("effective_from", ""), meta=meta)
        heads = list(SECTION_RE.finditer(body))
        if not heads:
            chunks.append(Chunk(id=common["doc_id"], section_title=doc_title, text=body.strip(), **common))
            continue
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
            text = body[h.end():end]
            text = re.split(r"^##\s", text, flags=re.M)[0].strip()  # stop at any untagged ## heading
            chunks.append(Chunk(id=h.group(1).strip(), section_title=h.group(2).strip(), text=text, **common))
    return chunks


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query_tokens):
        out = np.zeros(len(self.docs))
        for i, (d, L) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for t in query_tokens:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[i] = s
        return out


class Retriever:
    """Hybrid retriever. Embeddings are optional: if the model can't load, BM25 alone is used."""

    def __init__(self, use_embeddings: bool = True):
        self.chunks = load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        texts = [self._index_text(c) for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.model, self.emb = None, None
        if use_embeddings:
            try:
                self._load_embeddings(texts)
            except Exception as e:  # offline / missing package -> keyword search only
                print(f"[kb] embeddings unavailable ({e.__class__.__name__}: {e}); using BM25 only")

    @staticmethod
    def _index_text(c: Chunk) -> str:
        return f"{c.doc_title}. {c.section_title}. {c.id}. {c.text}"

    def _load_embeddings(self, texts):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL)
        key = hashlib.sha1((EMBED_MODEL + "\x00".join(texts)).encode()).hexdigest()[:16]
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"kb_emb_{key}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self._encode([f"passage: {t}" for t in texts])
            np.save(path, self.emb)

    def _encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=16))

    def search(self, query: str, k: int = TOP_K, include_superseded: bool = True):
        """Return [(chunk, score)] best first."""
        query = expand_query(query)
        q_tokens = tokenize(query)
        bm = self.bm25.scores(q_tokens)
        bm = bm / bm.max() if bm.max() > 0 else bm
        if self.emb is not None:
            qv = self._encode([f"query: {query}"])[0]
            cos = self.emb @ qv
            cos = (cos - cos.min()) / (cos.max() - cos.min() + 1e-9)
            score = 0.55 * cos + 0.45 * bm
        else:
            score = bm.copy()
        # Precedence re-ranking: small nudge so policy outranks FAQ/marketing on near-ties,
        # superseded docs only surface when clearly the best match.
        for i, c in enumerate(self.chunks):
            score[i] += 0.08 * (4 - c.rank) / 4
            if c.status == "superseded":
                score[i] -= 0.12 if include_superseded else 10
        order = np.argsort(-score)[:k]
        return [(self.chunks[i], float(score[i])) for i in order if score[i] > 0]


def chunks_to_json(hits):
    return [{"id": c.id, "doc": c.doc_file, "type": c.doc_type, "status": c.status,
             "score": round(s, 3), "text": c.text} for c, s in hits]


if __name__ == "__main__":
    import sys
    r = Retriever()
    print(f"{len(r.chunks)} chunks from {len({c.doc_file for c in r.chunks})} docs")
    for c, s in r.search(" ".join(sys.argv[1:]) or "refund to wallet how fast"):
        print(f"{s:.3f}  {c.header()}")
