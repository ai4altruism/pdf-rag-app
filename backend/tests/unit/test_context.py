"""Tests for parent-child context expansion and budget enforcement."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.pipelines.query.context import ContextAssembler
from src.storage.qdrant import SearchResult


@pytest.fixture
def mock_qdrant():
    return AsyncMock()


def _make_result(id_: str, score: float, text: str = "chunk text",
                 parent_id: str = "", chunk_index: int = 0) -> SearchResult:
    return SearchResult(
        id=id_,
        score=score,
        payload={
            "chunk_text": text,
            "parent_chunk_id": parent_id,
            "chunk_index": chunk_index,
            "document_id": "doc1",
            "filename": "test.pdf",
            "page_numbers": [1],
            "header_chain": ["Section 1"],
        },
    )


def _make_point(id_: str, payload: dict) -> SimpleNamespace:
    """Create a mock Qdrant point."""
    return SimpleNamespace(id=id_, payload=payload)


class TestContextAssemblerOff:
    async def test_off_mode_returns_results_unchanged(self, mock_qdrant):
        assembler = ContextAssembler(mock_qdrant, max_context_tokens=10000)
        results = [_make_result("c1", 0.9), _make_result("c2", 0.8)]

        output = await assembler.assemble(results, "col_1", mode="off")

        assert len(output) == 2
        assert output[0].id == "c1"

    async def test_empty_results(self, mock_qdrant):
        assembler = ContextAssembler(mock_qdrant)
        output = await assembler.assemble([], "col_1", mode="off")
        assert output == []


class TestContextAssemblerParent:
    async def test_replaces_children_with_parents(self, mock_qdrant):
        results = [
            _make_result("c1", 0.9, text="child 1", parent_id="p1", chunk_index=0),
            _make_result("c2", 0.8, text="child 2", parent_id="p1", chunk_index=1),
            _make_result("c3", 0.7, text="child 3", parent_id="p2", chunk_index=0),
        ]
        mock_qdrant.get_points_by_ids.return_value = [
            _make_point("p1", {"chunk_text": "parent 1 full text", "document_id": "doc1"}),
            _make_point("p2", {"chunk_text": "parent 2 full text", "document_id": "doc1"}),
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=100000)
        output = await assembler.assemble(results, "col_1", mode="parent")

        # Should have 2 parent results (deduplicated), not 3 children
        assert len(output) == 2
        # p1 gets best child score (0.9)
        assert output[0].id == "p1"
        assert output[0].score == 0.9
        assert output[1].id == "p2"

    async def test_keeps_results_without_parent(self, mock_qdrant):
        results = [
            _make_result("c1", 0.9, parent_id="p1"),
            _make_result("orphan", 0.8, parent_id=""),
        ]
        mock_qdrant.get_points_by_ids.return_value = [
            _make_point("p1", {"chunk_text": "parent text", "document_id": "doc1"}),
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=100000)
        output = await assembler.assemble(results, "col_1", mode="parent")

        ids = [r.id for r in output]
        assert "p1" in ids
        assert "orphan" in ids

    async def test_uses_best_child_score_for_parent(self, mock_qdrant):
        results = [
            _make_result("c1", 0.5, parent_id="p1"),
            _make_result("c2", 0.9, parent_id="p1"),
        ]
        mock_qdrant.get_points_by_ids.return_value = [
            _make_point("p1", {"chunk_text": "parent", "document_id": "doc1"}),
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=100000)
        output = await assembler.assemble(results, "col_1", mode="parent")

        assert output[0].score == 0.9


class TestContextAssemblerSiblings:
    async def test_adds_adjacent_siblings(self, mock_qdrant):
        results = [
            _make_result("c2", 0.9, text="chunk 2", parent_id="p1", chunk_index=2),
        ]
        mock_qdrant.get_points_by_filter.return_value = [
            _make_point("c1", {"chunk_text": "chunk 1", "chunk_index": 1, "parent_chunk_id": "p1"}),
            _make_point("c2", {"chunk_text": "chunk 2", "chunk_index": 2, "parent_chunk_id": "p1"}),
            _make_point("c3", {"chunk_text": "chunk 3", "chunk_index": 3, "parent_chunk_id": "p1"}),
            _make_point("c4", {"chunk_text": "chunk 4", "chunk_index": 4, "parent_chunk_id": "p1"}),
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=100000)
        output = await assembler.assemble(results, "col_1", mode="siblings")

        ids = [r.id for r in output]
        assert "c2" in ids  # original
        assert "c1" in ids  # left sibling
        assert "c3" in ids  # right sibling
        assert "c4" not in ids  # too far away

    async def test_no_duplicates_in_siblings(self, mock_qdrant):
        results = [
            _make_result("c1", 0.9, parent_id="p1", chunk_index=1),
            _make_result("c2", 0.8, parent_id="p1", chunk_index=2),
        ]
        # c2 is a sibling of c1, and also in original results
        mock_qdrant.get_points_by_filter.return_value = [
            _make_point("c1", {"chunk_text": "t", "chunk_index": 1, "parent_chunk_id": "p1"}),
            _make_point("c2", {"chunk_text": "t", "chunk_index": 2, "parent_chunk_id": "p1"}),
            _make_point("c3", {"chunk_text": "t", "chunk_index": 3, "parent_chunk_id": "p1"}),
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=100000)
        output = await assembler.assemble(results, "col_1", mode="siblings")

        ids = [r.id for r in output]
        assert ids.count("c1") == 1
        assert ids.count("c2") == 1


class TestBudgetEnforcement:
    async def test_enforces_token_budget(self, mock_qdrant):
        # Each "word " is ~1 token. Create chunks with known token counts.
        results = [
            _make_result("c1", 0.9, text="word " * 100),  # ~100 tokens
            _make_result("c2", 0.8, text="word " * 100),  # ~100 tokens
            _make_result("c3", 0.7, text="word " * 100),  # ~100 tokens
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=250)
        output = await assembler.assemble(results, "col_1", mode="off")

        assert len(output) == 2  # 3rd chunk would exceed budget

    async def test_always_includes_first_result(self, mock_qdrant):
        results = [
            _make_result("c1", 0.9, text="word " * 500),  # exceeds budget alone
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=100)
        output = await assembler.assemble(results, "col_1", mode="off")

        assert len(output) == 1  # Still includes first even if over budget

    async def test_empty_text_zero_tokens(self, mock_qdrant):
        results = [
            _make_result("c1", 0.9, text=""),
            _make_result("c2", 0.8, text=""),
        ]

        assembler = ContextAssembler(mock_qdrant, max_context_tokens=10)
        output = await assembler.assemble(results, "col_1", mode="off")

        assert len(output) == 2
