"""pgvector 集成测试：需要 docker compose up -d db。没起库时自动跳过。

⚠️ 教训（2026-08-21 真实事故）：早期版本直接用开发库跑测试，
prune_missing 测试把 502 条语料全清了——集成测试必须用独立数据库。
"""

import psycopg
import pytest

from app.chunker import Chunk
from app.config import settings
from app.embeddings import FakeEmbedder
from app.store import PgStore

TEST_DSN = settings.db_dsn.rsplit("/", 1)[0] + "/rag_test"


async def _test_db_ready() -> bool:
    """开发库可达时，确保独立的 rag_test 库存在并返回其可用性。"""
    try:
        conn = await psycopg.AsyncConnection.connect(
            settings.db_dsn, connect_timeout=2, autocommit=True
        )
    except Exception:
        return False
    try:
        cur = await conn.execute("SELECT 1 FROM pg_database WHERE datname = 'rag_test'")
        if not await cur.fetchone():
            await conn.execute("CREATE DATABASE rag_test")
    finally:
        await conn.close()
    return True


@pytest.mark.asyncio
async def test_roundtrip_search():
    if not await _test_db_ready():
        pytest.skip("pgvector 未启动（docker compose up -d db）")
    store = PgStore(TEST_DSN, dim=settings.embed_dim)
    await store.init()
    emb = FakeEmbedder(dim=settings.embed_dim)
    chunks = [
        Chunk("t.md", 0, "Python 的 asyncio 是协作式调度"),
        Chunk("t.md", 1, "Java 使用抢占式多线程模型"),
    ]
    vecs = await emb.embed([c.text for c in chunks])
    await store.replace_source("t.md", chunks, vecs)
    # 用第一条原文检索，自己必须排第一且分数接近 1
    hits = await store.search(vecs[0], k=2)
    assert hits[0].content == chunks[0].text
    assert hits[0].score > 0.99
    assert await store.count() >= 2


@pytest.mark.asyncio
async def test_replace_source_removes_stale_chunks():
    """文件重跑后变短：旧尾巴 chunk 必须被清掉，不能留孤儿污染检索。"""
    if not await _test_db_ready():
        pytest.skip("pgvector 未启动（docker compose up -d db）")
    store = PgStore(TEST_DSN, dim=settings.embed_dim)
    await store.init()
    emb = FakeEmbedder(dim=settings.embed_dim)
    long_chunks = [Chunk("re.md", i, f"第一版内容{i}") for i in range(3)]
    await store.replace_source("re.md", long_chunks, await emb.embed([c.text for c in long_chunks]))
    short_chunks = [Chunk("re.md", 0, "第二版只剩一块")]
    await store.replace_source(
        "re.md", short_chunks, await emb.embed([c.text for c in short_chunks])
    )
    rows = [h for h in await store.fetch_all() if h.source == "re.md"]
    assert len(rows) == 1
    assert rows[0].content == "第二版只剩一块"
    # prune：语料目录里没有 re.md 时应被整体清除
    removed = await store.prune_missing(["其他.md"])
    assert removed >= 1
    assert not [h for h in await store.fetch_all() if h.source == "re.md"]
