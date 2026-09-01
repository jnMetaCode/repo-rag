"""pgvector 存储层。v1 用精确检索（语料千级，无需 ANN 索引）。

连接管理：psycopg_pool 连接池（min 1 / max 10），懒加载 + 锁防并发重复建池。
最早版本每次查询新建连接——本机看不出差别，但每次多付 TCP+认证+类型注册
的开销，并发下连接数不受控；eval 脚本 gather 并发查询时池的收益最明显。
"""

import asyncio
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from .chunker import Chunk

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    source text NOT NULL,
    chunk_index int NOT NULL,
    content text NOT NULL,
    embedding vector({dim}) NOT NULL,
    UNIQUE (source, chunk_index)
);
"""


@dataclass
class Hit:
    source: str
    chunk_index: int
    content: str
    score: float  # 余弦相似度（1 - 距离），越大越相关


async def _on_connect(conn: psycopg.AsyncConnection) -> None:
    """池新建物理连接时跑一次：注册 vector 类型，之后复用免注册。"""
    await register_vector_async(conn)


class PgStore:
    def __init__(self, dsn: str, dim: int) -> None:
        self._dsn = dsn
        self._dim = dim
        self._pool: AsyncConnectionPool | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> AsyncConnectionPool:
        """懒加载：eval 脚本不走 init() 直接查询，也要能拿到池。"""
        async with self._pool_lock:
            if self._pool is None:
                pool = AsyncConnectionPool(
                    self._dsn,
                    min_size=1,
                    max_size=10,
                    open=False,
                    configure=_on_connect,
                    kwargs={"autocommit": True},
                )
                await pool.open()
                self._pool = pool
        return self._pool

    async def init(self) -> None:
        # DDL 用一次性裸连接：池的 configure 要注册 vector 类型，
        # 扩展还没建时注册会失败——先建表再开池
        async with await psycopg.AsyncConnection.connect(self._dsn, autocommit=True) as conn:
            await conn.execute(_DDL.format(dim=self._dim))
        await self._get_pool()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def replace_source(
        self, source: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        """整文件替换：先删旧 chunk 再插新——文件变短/重新分块时不留孤儿。

        只 upsert 不删除是重跑型 ingest 的经典 bug：旧尾巴 chunk 永远留在库里污染检索。
        DELETE+INSERT 包在单事务里：重建期间的并发查询要么看到旧版要么看到新版，
        进程中途崩溃也不会留下删了旧数据、新数据只插了一半的空洞。
        """
        assert len(chunks) == len(embeddings)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.transaction(), conn.cursor() as cur:
                await cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
                for c, e in zip(chunks, embeddings, strict=True):
                    await cur.execute(
                        """
                        INSERT INTO chunks (source, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source, chunk_index)
                        DO UPDATE SET content = EXCLUDED.content,
                                      embedding = EXCLUDED.embedding
                        """,
                        (c.source, c.index, c.text, e),
                    )
        return len(chunks)

    async def search(self, query_vec: list[float], k: int) -> list[Hit]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT source, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vec, query_vec, k),
                )
                rows = await cur.fetchall()
        return [Hit(source=r[0], chunk_index=r[1], content=r[2], score=float(r[3])) for r in rows]

    async def prune_missing(self, keep_sources: list[str]) -> int:
        """删除已不存在于语料目录的文件对应的所有 chunk。返回删除行数。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chunks WHERE NOT (source = ANY(%s))", (keep_sources,)
                )
                return cur.rowcount

    async def fetch_all(self) -> list[Hit]:
        """全量拉取（构建内存 BM25 索引用；千级语料毫秒级）。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT source, chunk_index, content FROM chunks")
                rows = await cur.fetchall()
        return [Hit(source=r[0], chunk_index=r[1], content=r[2], score=0.0) for r in rows]

    async def count(self) -> int:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM chunks")
                row = await cur.fetchone()
        return int(row[0]) if row else 0
