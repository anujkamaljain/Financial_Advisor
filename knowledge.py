"""Vanilla RAG over the curated finance corpus. No vector database.

Chunks are the `##` sections of `data/knowledge.md`. Embedding with Gemini is
optional: the corpus is small enough that token overlap is a valid retriever,
and Cloud deploys cannot depend on a paid embedding call succeeding.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CACHE_DIR, DATA_DIR, EMBED_MODEL, GEMINI_API_KEY

CORPUS_FILE = DATA_DIR / "knowledge.md"
EMBED_CACHE = CACHE_DIR / "embeddings.npz"
_TOKEN = re.compile(r"[a-z0-9]+")


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
    """Prefer cached Gemini embeddings; fall back to lexical overlap."""

    def __init__(self) -> None:
        self.chunks = load_chunks()
        self.headings = [h for h, _ in self.chunks]
        self.bodies = [b for _, b in self.chunks]
        self._embeddings: GoogleGenerativeAIEmbeddings | None = None
        self.matrix = self._load_or_build()

    def _corpus_hash(self) -> str:
        joined = "\u241e".join(f"{h}\u241f{b}" for h, b in self.chunks) + EMBED_MODEL
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _client(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBED_MODEL, google_api_key=GEMINI_API_KEY
            )
        return self._embeddings

    def _load_or_build(self) -> np.ndarray | None:
        digest = self._corpus_hash()
        if EMBED_CACHE.exists():
            try:
                cached = np.load(EMBED_CACHE, allow_pickle=False)
                if str(cached["digest"].item()) == digest:
                    return cached["matrix"]
            except (OSError, ValueError, KeyError):
                pass

        try:
            vectors = self._client().embed_documents(
                [f"{h}. {b}" for h, b in self.chunks]
            )
        except Exception:  # noqa: BLE001 - quota/billing must not take the app down
            return None
        matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(EMBED_CACHE, matrix=matrix, digest=np.array(digest))
        return matrix

    def _lexical(self, query: str, k: int) -> list[tuple[str, str, float]]:
        query_tokens = set(_TOKEN.findall(query.lower()))
        scored: list[tuple[str, str, float]] = []
        for heading, body in self.chunks:
            tokens = set(_TOKEN.findall(f"{heading} {body}".lower()))
            score = (len(query_tokens & tokens) / len(query_tokens)) if query_tokens else 0.0
            scored.append((heading, body, score))
        scored.sort(key=lambda row: -row[2])
        return scored[:k]

    def search(self, query: str, k: int = 4) -> list[tuple[str, str, float]]:
        """Return the k most similar chunks as (heading, body, score)."""
        if self.matrix is not None:
            try:
                vector = _normalise(
                    np.asarray([self._client().embed_query(query)], dtype=np.float32)
                )[0]
                scores = self.matrix @ vector
                top = np.argsort(-scores)[:k]
                return [(self.headings[i], self.bodies[i], float(scores[i])) for i in top]
            except Exception:  # noqa: BLE001 - embeddings are optional
                pass
        return self._lexical(query, k)

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
