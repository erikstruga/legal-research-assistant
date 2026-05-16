"""
RAG pipeline using LangChain + ChromaDB + GPT-4o.
"""

import logging
from typing import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.core.embeddings import get_vector_store
from app.models.schemas import ChatMessage, SourceDocument
from app.services.courtlistener import CaseLawDocument
from app.services.congress import StatuteDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable legal research assistant. You help users \
understand case law, statutes, and legal concepts by summarizing and explaining \
documents retrieved from authoritative public legal databases.

Guidelines:
- Always cite the source documents you reference.
- Clearly distinguish between case law (judicial decisions) and statutes (legislation).
- Explain legal jargon in plain language when helpful.
- Never provide legal advice — clarify that your answers are for research and informational purposes only.
- If the retrieved context does not contain enough information to answer the question, say so clearly.

Context from legal databases:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


class RAGPipeline:
    def __init__(self):
        settings = get_settings()
        self._settings = settings
        self._vector_store = get_vector_store()
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            streaming=True,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def _doc_to_langchain(
        self, raw_doc: CaseLawDocument | StatuteDocument
    ) -> list[Document]:
        """Convert a raw API document into chunked LangChain Documents."""
        if isinstance(raw_doc, CaseLawDocument):
            content = raw_doc.plain_text
            metadata = {
                "title": raw_doc.title,
                "doc_type": "case_law",
                "court": raw_doc.court,
                "date_filed": raw_doc.date_filed or "",
                "url": raw_doc.url,
                "source": "CourtListener",
            }
        else:
            content = raw_doc.summary
            metadata = {
                "title": raw_doc.full_title,
                "doc_type": "statute",
                "congress": str(raw_doc.congress),
                "url": raw_doc.url,
                "source": "Congress.gov",
            }

        chunks = self._splitter.create_documents([content], metadatas=[metadata])
        return chunks

    def ingest_documents(
        self, raw_docs: list[CaseLawDocument | StatuteDocument]
    ) -> int:
        """Chunk and embed raw documents into the vector store. Returns total chunks added."""
        all_chunks: list[Document] = []
        for doc in raw_docs:
            chunks = self._doc_to_langchain(doc)
            all_chunks.extend(chunks)

        if all_chunks:
            self._vector_store.add_documents(all_chunks)
            logger.info(f"Ingested {len(all_chunks)} chunks from {len(raw_docs)} documents")

        return len(all_chunks)

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def _retrieve(self, question: str, source_filter: str | None = None) -> list[Document]:
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": self._settings.retriever_k}
        )
        docs = retriever.invoke(question)

        if source_filter and source_filter != "all":
            docs = [d for d in docs if d.metadata.get("doc_type") == source_filter]

        return docs

    def _format_context(self, docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('title', 'Unknown')} ({meta.get('source', '')})"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _docs_to_sources(self, docs: list[Document]) -> list[SourceDocument]:
        seen = set()
        sources = []
        for doc in docs:
            meta = doc.metadata
            key = meta.get("url") or meta.get("title")
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceDocument(
                    title=meta.get("title", "Unknown"),
                    source=meta.get("source", ""),
                    url=meta.get("url"),
                    snippet=doc.page_content[:300] + "…",
                    doc_type=meta.get("doc_type", "case_law"),
                )
            )
        return sources

    # ─── Chat ─────────────────────────────────────────────────────────────────

    def _build_history(self, history: list[ChatMessage]):
        messages = []
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        return messages

    async def chat(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        source_filter: str = "all",
    ) -> tuple[str, list[SourceDocument]]:
        """Run a full RAG cycle and return (answer, sources)."""
        docs = self._retrieve(question, source_filter)
        context = self._format_context(docs)
        sources = self._docs_to_sources(docs)
        lc_history = self._build_history(history or [])

        chain = RAG_PROMPT | self._llm | StrOutputParser()

        answer = await chain.ainvoke(
            {
                "context": context,
                "history": lc_history,
                "question": question,
            }
        )
        return answer, sources

    async def stream_chat(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        source_filter: str = "all",
    ) -> AsyncIterator[str]:
        """Stream the RAG answer token by token."""
        docs = self._retrieve(question, source_filter)
        context = self._format_context(docs)
        lc_history = self._build_history(history or [])

        chain = RAG_PROMPT | self._llm | StrOutputParser()
        async for chunk in chain.astream(
            {
                "context": context,
                "history": lc_history,
                "question": question,
            }
        ):
            yield chunk

    def vector_store_count(self) -> int:
        try:
            return self._vector_store._collection.count()
        except Exception:
            return -1


# Singleton
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
