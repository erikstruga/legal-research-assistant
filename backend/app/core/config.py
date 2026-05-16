from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # CourtListener
    courtlistener_api_token: str = ""
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v3"

    # Congress.gov
    congress_api_key: str = ""
    congress_base_url: str = "https://api.congress.gov/v3"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "legal_docs"

    # RAG
    retriever_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # API
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
