from pydantic import BaseModel, Field
from typing import Literal, Optional


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    source_filter: Optional[Literal["case_law", "statutes", "all"]] = "all"


class SourceDocument(BaseModel):
    title: str
    source: str
    url: Optional[str] = None
    snippet: str
    doc_type: Literal["case_law", "statute"]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    question: str


# ─── Search / Ingest ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    source: Literal["courtlistener", "congress", "both"] = "both"
    max_results: int = Field(default=10, ge=1, le=50)
    ingest: bool = Field(default=True, description="Ingest results into the vector store")


class IngestResult(BaseModel):
    source: str
    title: str
    url: Optional[str]
    chunks_added: int


class SearchResponse(BaseModel):
    query: str
    results_ingested: list[IngestResult]
    total_chunks: int


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store_docs: int
