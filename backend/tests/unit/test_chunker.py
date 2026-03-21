"""Tests for document-aware chunker with parent-child relationships."""

import pytest

from src.pipelines.ingestion.chunker import DocumentChunker


class TestDocumentChunker:
    """Tests for the DocumentChunker class."""

    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size_tokens=50, chunk_overlap_tokens=10)

    def test_empty_text_returns_no_chunks(self, chunker):
        chunks = chunker.chunk("")
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self, chunker):
        chunks = chunker.chunk("   \n\n  ")
        assert chunks == []

    def test_simple_text_without_headers(self, chunker):
        """Text without headers produces one parent and one child."""
        chunks = chunker.chunk("This is a simple paragraph.")
        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        assert len(parents) == 1
        assert len(children) == 1
        assert children[0].parent_chunk_id == parents[0].chunk_id
        assert children[0].header_chain == []

    def test_header_splitting(self, chunker):
        """Headers create separate sections with header chains."""
        md = "# Introduction\n\nFirst section.\n\n## Details\n\nSecond section."
        chunks = chunker.chunk(md)
        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        assert len(parents) == 2
        assert len(children) == 2

        # First section
        assert children[0].header_chain == ["Introduction"]
        # Second section inherits parent header
        assert children[1].header_chain == ["Introduction", "Details"]

    def test_h3_headers(self, chunker):
        """H3 headers are split and tracked in chain."""
        md = "# Top\n\nA\n\n## Mid\n\nB\n\n### Deep\n\nC"
        chunks = chunker.chunk(md)
        children = [c for c in chunks if not c.is_parent]

        assert children[0].header_chain == ["Top"]
        assert children[1].header_chain == ["Top", "Mid"]
        assert children[2].header_chain == ["Top", "Mid", "Deep"]

    def test_preamble_before_first_header(self, chunker):
        """Text before the first header becomes its own section."""
        md = "Preamble text.\n\n# First Header\n\nContent."
        chunks = chunker.chunk(md)
        children = [c for c in chunks if not c.is_parent]

        assert len(children) == 2
        assert "Preamble" in children[0].text
        assert children[0].header_chain == []

    def test_parent_child_relationship_integrity(self, chunker):
        """Every child references an existing parent."""
        md = "# A\n\nContent A.\n\n## B\n\nContent B.\n\n# C\n\nContent C."
        chunks = chunker.chunk(md)
        parent_ids = {c.chunk_id for c in chunks if c.is_parent}
        children = [c for c in chunks if not c.is_parent]

        for child in children:
            assert child.parent_chunk_id in parent_ids

    def test_unique_chunk_ids(self, chunker):
        """All chunks have unique IDs."""
        md = "# A\n\nContent.\n\n# B\n\nMore content.\n\n# C\n\nEven more."
        chunks = chunker.chunk(md)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_monotonic(self, chunker):
        """Child chunk indices are monotonically increasing."""
        md = "# A\n\nContent A.\n\n## B\n\nContent B.\n\n# C\n\nContent C."
        chunks = chunker.chunk(md)
        children = [c for c in chunks if not c.is_parent]
        indices = [c.chunk_index for c in children]
        assert indices == sorted(indices)
        assert indices == list(range(len(children)))

    def test_page_numbers_passed_through(self, chunker):
        """Page numbers are assigned to all chunks."""
        md = "# Test\n\nSome content."
        chunks = chunker.chunk(md, page_numbers=[1, 2, 3])
        for chunk in chunks:
            assert chunk.page_numbers == [1, 2, 3]

    def test_token_count_populated(self, chunker):
        """All chunks have a positive token count."""
        md = "# Header\n\nSome content goes here."
        chunks = chunker.chunk(md)
        for chunk in chunks:
            assert chunk.token_count > 0


class TestChunkerTokenSplitting:
    """Tests for the token-level splitting behavior of oversized chunks."""

    @pytest.fixture
    def small_chunker(self):
        """Chunker with very small chunk size to force splitting."""
        return DocumentChunker(chunk_size_tokens=20, chunk_overlap_tokens=5)

    def test_oversized_section_produces_multiple_children(self, small_chunker):
        """A section exceeding chunk_size_tokens is split into multiple children."""
        # ~100+ tokens
        long_text = "# Header\n\n" + " ".join(["word"] * 100)
        chunks = small_chunker.chunk(long_text)
        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        assert len(parents) == 1
        assert len(children) > 1

        # All children reference the same parent
        for child in children:
            assert child.parent_chunk_id == parents[0].chunk_id

    def test_child_tokens_within_limit(self, small_chunker):
        """Each child chunk is within the token limit (approximately)."""
        long_text = "# Header\n\n" + " ".join(["word"] * 100)
        chunks = small_chunker.chunk(long_text)
        children = [c for c in chunks if not c.is_parent]

        for child in children:
            # Allow some slack for sentence-boundary breaking
            assert child.token_count <= small_chunker._chunk_size * 1.3

    def test_all_text_preserved(self, small_chunker):
        """Splitting doesn't lose significant content."""
        words = [f"word{i}" for i in range(50)]
        long_text = " ".join(words)
        chunks = small_chunker.chunk(long_text)
        children = [c for c in chunks if not c.is_parent]

        # All original words should appear in at least one child
        combined = " ".join(c.text for c in children)
        for word in words:
            assert word in combined


class TestChunkerHeaderChainReset:
    """Tests for header chain behavior across sibling headers."""

    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)

    def test_sibling_h2_resets_chain(self, chunker):
        """A second H2 under the same H1 replaces the previous H2 in the chain."""
        md = "# Top\n\nIntro\n\n## A\n\nContent A\n\n## B\n\nContent B"
        chunks = chunker.chunk(md)
        children = [c for c in chunks if not c.is_parent]

        # Children should have:
        # ["Top"] for intro, ["Top", "A"], ["Top", "B"]
        chains = [c.header_chain for c in children]
        assert ["Top"] in chains
        assert ["Top", "A"] in chains
        assert ["Top", "B"] in chains
        # "A" should not appear in B's chain
        b_chunk = next(c for c in children if "Content B" in c.text)
        assert "A" not in b_chunk.header_chain

    def test_new_h1_resets_all(self, chunker):
        """A new H1 resets the entire header chain."""
        md = "# First\n\n## Sub\n\nA\n\n# Second\n\nB"
        chunks = chunker.chunk(md)
        children = [c for c in chunks if not c.is_parent]

        second_chunk = next(c for c in children if "Second" in c.text or "B" in c.text)
        assert "Sub" not in second_chunk.header_chain
