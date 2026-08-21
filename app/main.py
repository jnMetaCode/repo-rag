"""FastAPI 入口。入库走 CLI（app/ingest.py），HTTP 只读——无鉴权的写接口是事故。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import settings
from .embeddings import make_embedder
from .keyword import Bm25Index
from .llm import make_llm
from .rag import answer
from .store import PgStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    client = httpx.AsyncClient()
    app.state.store = PgStore(settings.db_dsn, settings.embed_dim)
    await app.state.store.init()
    app.state.embedder = make_embedder(client, settings)
    app.state.llm = make_llm(client, settings)
    app.state.bm25 = Bm25Index()
    if settings.hybrid:
        app.state.bm25.build(await app.state.store.fetch_all())
    try:
        yield
    finally:
        await client.aclose()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


def create_app() -> FastAPI:
    app = FastAPI(title="repo-rag", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        from .ui import PAGE

        return PAGE

    @app.post("/v1/query")
    async def query(req: QueryRequest, request: Request) -> dict:
        result = await answer(
            req.question,
            embedder=request.app.state.embedder,
            store=request.app.state.store,
            llm=request.app.state.llm,
            top_k=req.top_k,
            min_score=settings.min_score,
            bm25=request.app.state.bm25 if settings.hybrid else None,
        )
        return {
            "answer": result.answer,
            "sources": result.sources,
            "refused": result.refused,
            "top_score": round(result.top_score, 4),
        }

    @app.get("/v1/stats")
    async def stats(request: Request) -> dict:
        return {
            "chunks": await request.app.state.store.count(),
            "bm25_indexed": request.app.state.bm25.size,
            "retrieval": "hybrid(rrf)" if settings.hybrid else "vector",
        }

    @app.post("/v1/reindex")
    async def reindex(request: Request) -> dict:
        """ingest 之后重建内存 BM25 索引（免重启）。只读 DB，无副作用。"""
        request.app.state.bm25.build(await request.app.state.store.fetch_all())
        return {"bm25_indexed": request.app.state.bm25.size}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
