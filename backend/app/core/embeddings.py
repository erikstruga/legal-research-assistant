import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from functools import lru_cache

from app.core.config import get_settings


@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_vector_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        client=get_chroma_client(),
    )
