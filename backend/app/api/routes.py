from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    IngestResult,
    HealthResponse,
)
from app.services.rag import get_pipeline
from app.services.courtlistener import CourtListenerClient
from app.services.congress import CongressClient

router = APIRouter()


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        vector_store_docs=pipeline.vector_store_count(),
    )


# ─── Chat ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a legal research question. Returns answer + cited sources."""
    pipeline = get_pipeline()
    try:
        answer, sources = await pipeline.chat(
            question=req.question,
            history=req.history,
            source_filter=req.source_filter or "all",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(answer=answer, sources=sources, question=req.question)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming version of /chat.
    Returns an SSE stream: each event is a JSON object with 'type' and 'data'.
    Types: 'token' | 'sources' | 'done' | 'error'
    """
    pipeline = get_pipeline()

    async def event_stream():
        try:
            # First retrieve docs and emit sources
            from app.services.rag import RAGPipeline
            docs = pipeline._retrieve(req.question, req.source_filter or "all")
            sources = pipeline._docs_to_sources(docs)
            sources_payload = [s.model_dump() for s in sources]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_payload})}\n\n"

            # Stream tokens
            async for token in pipeline.stream_chat(
                question=req.question,
                history=req.history,
                source_filter=req.source_filter or "all",
            ):
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Search & Ingest ─────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def search_and_ingest(req: SearchRequest):
    """
    Search CourtListener and/or Congress.gov, optionally ingesting results
    into the vector store so they become available for RAG queries.
    """
    pipeline = get_pipeline()
    ingested: list[IngestResult] = []
    total_chunks = 0

    cl_client = CourtListenerClient()
    cg_client = CongressClient()

    try:
        tasks = []
        if req.source in ("courtlistener", "both"):
            tasks.append(("courtlistener", cl_client.search_opinions(req.query, req.max_results)))
        if req.source in ("congress", "both"):
            tasks.append(("congress", cg_client.search_bills(req.query, req.max_results)))

        for source_name, coro in tasks:
            docs = await coro
            if req.ingest:
                for doc in docs:
                    chunks = pipeline._doc_to_langchain(doc)
                    if chunks:
                        pipeline._vector_store.add_documents(chunks)
                        ingested.append(
                            IngestResult(
                                source=source_name,
                                title=getattr(doc, "title", None) or getattr(doc, "full_title", ""),
                                url=getattr(doc, "url", None),
                                chunks_added=len(chunks),
                            )
                        )
                        total_chunks += len(chunks)

    finally:
        await cl_client.close()
        await cg_client.close()

    return SearchResponse(
        query=req.query,
        results_ingested=ingested,
        total_chunks=total_chunks,
    )
