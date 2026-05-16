from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from app.core.config import get_settings
from app.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Legal Research Assistant API...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Legal Research Assistant",
    description=(
        "A RAG-based chatbot that searches and summarizes case law and statutes "
        "from CourtListener and Congress.gov."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Serve frontend static files if present
import os
if os.path.isdir("/app/frontend"):
    app.mount("/", StaticFiles(directory="/app/frontend", html=True), name="frontend")
