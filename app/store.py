"""pgvector 存储层。v1 用精确检索（语料千级，无需 ANN 索引）。"""

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector_async

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


class PgStore:
    def __init__(self, dsn: str, dim: int) -> None:
        self._dsn = dsn
        self._dim = dim

    async def _conn(self) -> psycopg.AsyncConnection:
        conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        await register_vector_async(conn)
        return conn

    async def init(self) -> None:
        async with await psycopg.AsyncConnection.connect(self._dsn, autocommit=True) as conn:
            await conn.execute(_DDL.format(dim=self._dim))

    async def replace_source(
        self, source: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        """整文件替换：先删旧 chunk 再插新——文件变短/重新分块时不留孤儿。

        只 upsert 不删除是重跑型 ingest 的经典 bug：旧尾巴 chunk 永远留在库里污染检索。
        """
        assert len(chunks) == len(embeddings)
        async with await self._conn() as conn:
            async with conn.cursor() as cur:
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
        async with await self._conn() as conn:
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
        async with await self._conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chunks WHERE NOT (source = ANY(%s))", (keep_sources,)
                )
                return cur.rowcount

    async def fetch_all(self) -> list[Hit]:
        """全量拉取（构建内存 BM25 索引用；千级语料毫秒级）。"""
        async with await self._conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT source, chunk_index, content FROM chunks")
                rows = await cur.fetchall()
        return [Hit(source=r[0], chunk_index=r[1], content=r[2], score=0.0) for r in rows]

    async def count(self) -> int:
        async with await self._conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM chunks")
                row = await cur.fetchone()
        return int(row[0]) if row else 0
