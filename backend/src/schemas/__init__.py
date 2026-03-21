"""Pydantic request/response schemas for the API."""

from datetime import datetime

from pydantic import BaseModel, Field


# --- Collections ---
class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class CollectionUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str | None
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Documents ---
class DocumentResponse(BaseModel):
    id: str
    collection_id: str
    filename: str
    title: str | None
    author: str | None
    page_count: int | None
    file_size_bytes: int | None
    status: str
    parser_used: str | None
    chunk_count: int | None
    embedding_model: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReingestRequest(BaseModel):
    parser: str | None = None
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None
    contextual_enrichment: bool | None = None


# --- Query ---
class QueryOptions(BaseModel):
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    multi_query: bool = True
    corrective_rag: bool = True
    context_expansion: str = "parent"
    max_context_tokens: int = 8000


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_ids: list[str] = Field(..., min_length=1)
    conversation_id: str | None = None
    options: QueryOptions = Field(default_factory=QueryOptions)


class SourceResponse(BaseModel):
    document_id: str
    filename: str
    page_numbers: list[int]
    header_chain: list[str]
    chunk_text: str
    relevance_score: float


class QueryMetadata(BaseModel):
    retrieval_count: int
    reranked_count: int
    corrective_rag_triggered: bool
    query_variations: list[str]
    latency_ms: int
    model_used: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    metadata: QueryMetadata


# --- Conversations ---
class ConversationCreate(BaseModel):
    collection_id: str
    title: str = "New Conversation"


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: str
    collection_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []

    model_config = {"from_attributes": True}


# --- Settings ---
class SettingsResponse(BaseModel):
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    llm_provider: str
    llm_model: str
    reranker_provider: str
    reranker_model: str
    parser: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    contextual_enrichment: bool
    top_k_retrieval: int
    top_k_rerank: int
    hybrid_search: bool
    multi_query: bool
    corrective_rag: bool
    context_expansion: str
    max_context_tokens: int


# --- Health ---
class DependencyStatus(BaseModel):
    status: str  # "ok" or "error"
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: DependencyStatus
    embedding_provider: DependencyStatus
    llm_provider: DependencyStatus
