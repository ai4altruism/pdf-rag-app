"""Tests for BM25 sparse vector encoder."""

from src.storage.sparse import BM25SparseEncoder, SparseVector


class TestSparseVector:
    def test_sparse_vector_dataclass(self):
        sv = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        assert sv.indices == [1, 2, 3]
        assert sv.values == [0.5, 0.3, 0.2]

    def test_sparse_vector_empty(self):
        sv = SparseVector(indices=[], values=[])
        assert sv.indices == []
        assert sv.values == []


class TestBM25SparseEncoder:
    def test_encode_simple_text(self):
        encoder = BM25SparseEncoder()
        result = encoder.encode("machine learning algorithms")
        assert isinstance(result, SparseVector)
        assert len(result.indices) > 0
        assert len(result.indices) == len(result.values)

    def test_encode_empty_text(self):
        encoder = BM25SparseEncoder()
        result = encoder.encode("")
        assert result.indices == []
        assert result.values == []

    def test_encode_stopwords_only(self):
        encoder = BM25SparseEncoder()
        result = encoder.encode("the is a an")
        assert result.indices == []
        assert result.values == []

    def test_encode_deterministic(self):
        encoder = BM25SparseEncoder()
        r1 = encoder.encode("deep neural network")
        r2 = encoder.encode("deep neural network")
        assert r1.indices == r2.indices
        assert r1.values == r2.values

    def test_encode_indices_sorted(self):
        encoder = BM25SparseEncoder()
        result = encoder.encode("zebra apple mango banana cherry")
        assert result.indices == sorted(result.indices)

    def test_encode_values_positive(self):
        encoder = BM25SparseEncoder()
        result = encoder.encode("retrieval augmented generation")
        assert all(v > 0 for v in result.values)

    def test_encode_filters_short_tokens(self):
        """Single-character tokens should be filtered out."""
        encoder = BM25SparseEncoder()
        result = encoder.encode("I x y data")
        # "i", "x", "y" should be filtered (len <= 1); "data" should remain
        assert len(result.indices) == 1

    def test_encode_case_insensitive(self):
        encoder = BM25SparseEncoder()
        r1 = encoder.encode("Machine Learning")
        r2 = encoder.encode("machine learning")
        assert r1.indices == r2.indices
        assert r1.values == r2.values

    def test_encode_repeated_terms_higher_weight(self):
        encoder = BM25SparseEncoder()
        r_once = encoder.encode("neural")
        r_twice = encoder.encode("neural neural")
        # The repeated term should have a higher BM25 weight
        assert r_twice.values[0] > r_once.values[0]

    def test_encode_batch(self):
        encoder = BM25SparseEncoder()
        texts = ["hello world", "machine learning", "deep neural network"]
        results = encoder.encode_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, SparseVector) for r in results)

    def test_encode_batch_empty(self):
        encoder = BM25SparseEncoder()
        results = encoder.encode_batch([])
        assert results == []

    def test_vocab_size_respected(self):
        encoder = BM25SparseEncoder(vocab_size=100)
        result = encoder.encode("some random text with various words here")
        assert all(0 <= idx < 100 for idx in result.indices)

    def test_hash_token_deterministic(self):
        encoder = BM25SparseEncoder()
        h1 = encoder._hash_token("hello")
        h2 = encoder._hash_token("hello")
        assert h1 == h2

    def test_hash_token_within_vocab(self):
        encoder = BM25SparseEncoder(vocab_size=500)
        for word in ["cat", "dog", "bird", "fish", "python"]:
            assert 0 <= encoder._hash_token(word) < 500
