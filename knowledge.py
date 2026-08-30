"""Vanilla RAG over the curated finance corpus. No vector database.

Chunks are the `##` sections of `data/knowledge.md`, embedded once with Gemini
and cached to disk; retrieval is a numpy cosine similarity over a small matrix.
At this corpus size a vector store would be pure ceremony -- an exact search
over a few dozen vectors is both faster and easier to reason about.

RAG is used here for the *advice policy* (what good practice looks like), while
the research crew supplies the *live numbers*. Keeping those two sources of
truth separate is what stops the model from blending stale text into fresh rates.
"""

from __future__ import annotations

import hashlib

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CACHE_DIR, DATA_DIR, EMBED_MODEL, GEMINI_API_KEY

CORPUS_FILE = DATA_DIR / "knowledge.md"
EMBED_CACHE = CACHE_DIR / "embeddings.npz"


def load_chunks() -> list[tuple[str, str]]:
    """Split the corpus into (heading, body) pairs."""
    text = CORPUS_FILE.read_text(encoding="utf-8")
    chunks: list[tuple[str, str]] = []
    for block in text.split("\n## ")[1:]:
        heading, _, body = block.partition("\n")
        body = body.strip()
        if body:
            chunks.append((heading.strip(), body))
    return chunks


class KnowledgeBase:
    """Embed-once, search-many retriever."""

    def __init__(self) -> None:
        self.chunks = load_chunks()
        self.headings = [h for h, _ in self.chunks]
        self.bodies = [b for _, b in self.chunks]
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL, google_api_key=GEMINI_API_KEY
        )
        self.matrix = self._load_or_build()

    def _corpus_hash(self) -> str:
        joined = "\u241e".join(f"{h}\u241f{b}" for h, b in self.chunks) + EMBED_MODEL
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _load_or_build(self) -> np.ndarray:
        digest = self._corpus_hash()
        if EMBED_CACHE.exists():
            cached = np.load(EMBED_CACHE, allow_pickle=False)
            if str(cached["digest"].item()) == digest:
                return cached["matrix"]

        vectors = self._embeddings.embed_documents(
            [f"{h}. {b}" for h, b in self.chunks]
        )
        matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(EMBED_CACHE, matrix=matrix, digest=np.array(digest))
        return matrix

    def search(self, query: str, k: int = 4) -> list[tuple[str, str, float]]:
        """Return the k most similar chunks as (heading, body, score)."""
        vector = _normalise(
            np.asarray([self._embeddings.embed_query(query)], dtype=np.float32)
        )[0]
        scores = self.matrix @ vector
        top = np.argsort(-scores)[:k]
        return [(self.headings[i], self.bodies[i], float(scores[i])) for i in top]

    def context_for(self, query: str, k: int = 4) -> str:
        """Retrieved chunks formatted for injection into a prompt."""
        return "\n\n".join(
            f"### {heading}\n{body}" for heading, body, _ in self.search(query, k)
        )


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


_INSTANCE: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Lazy singleton so the embedding cost is paid at most once per process."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = KnowledgeBase()
    return _INSTANCE
