"""BM25-based sparse vector encoder for hybrid search."""

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class SparseVector:
    """Sparse vector representation with term indices and weights."""

    indices: list[int]
    values: list[float]


class BM25SparseEncoder:
    """Encodes text into sparse BM25-style vectors for Qdrant sparse search.

    Uses a simple hash-based vocabulary (no pre-built IDF corpus needed).
    Term frequencies are weighted with BM25 saturation (k1 parameter).
    """

    def __init__(self, vocab_size: int = 30000, k1: float = 1.2, b: float = 0.75):
        self._vocab_size = vocab_size
        self._k1 = k1
        self._b = b
        self._avg_dl = 256.0  # assumed average document length in tokens

    def encode(self, text: str) -> SparseVector:
        """Encode text into a sparse vector."""
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf = Counter(tokens)
        dl = len(tokens)

        indices = []
        values = []

        for token, count in tf.items():
            idx = self._hash_token(token)
            # BM25 term frequency saturation
            tf_score = (count * (self._k1 + 1)) / (
                count + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
            )
            indices.append(idx)
            values.append(round(tf_score, 4))

        # Sort by index for Qdrant
        pairs = sorted(zip(indices, values, strict=True))
        return SparseVector(
            indices=[p[0] for p in pairs],
            values=[p[1] for p in pairs],
        )

    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """Encode multiple texts into sparse vectors."""
        return [self.encode(text) for text in texts]

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        # Filter very short tokens and stopwords
        return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]

    def _hash_token(self, token: str) -> int:
        """Hash a token to a vocab index. Deterministic across runs."""
        h = 0
        for ch in token:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h % self._vocab_size


# Minimal English stopword set
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "with", "as", "at", "by", "be", "was", "were", "been", "are",
    "this", "that", "from", "not", "but", "had", "has", "have", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "if", "so", "no", "up", "out", "we", "he", "she",
    "they", "you", "my", "his", "her", "its", "our", "your", "their",
    "me", "him", "us", "them", "who", "what", "which", "when", "where",
    "how", "than", "then", "just", "also", "into", "over", "such",
    "after", "before", "between", "each", "all", "any", "both", "few",
    "more", "most", "other", "some", "about", "only", "very",
})
