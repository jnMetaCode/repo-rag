"""入库 CLI：python -m app.ingest data/

扫描目录下 *.md -> 分块 -> 批量 embed -> upsert 到 pgvector。
全量重跑即更新（v1 明确不做增量）。

文件 IO 在进入 async 之前同步完成（CLI 一次性任务，
不能把阻塞的 pathlib 调用混进事件循环——ruff ASYNC240 拦的就是这个）。
"""

import asyncio
import sys
from pathlib import Path

import httpx

from .chunker import chunk_by_strategy
from .config import settings
from .embeddings import make_embedder
from .store import PgStore


def load_files(directory: Path) -> list[tuple[str, str]]:
    """同步读盘：返回 [(文件名, 文本)]。"""
    return [
        (f.name, f.read_text(encoding="utf-8", errors="replace"))
        for f in sorted(directory.rglob("*.md"))
    ]


async def ingest(files: list[tuple[str, str]]) -> None:
    store = PgStore(settings.db_dsn, settings.embed_dim)
    await store.init()
    try:
        await _ingest_all(store, files)
    finally:
        await store.close()


async def _ingest_all(store: PgStore, files: list[tuple[str, str]]) -> None:
    async with httpx.AsyncClient() as client:
        embedder = make_embedder(client, settings)
        total = 0
        for name, text in files:
            chunks = chunk_by_strategy(
                settings.chunk_strategy,
                name,
                text,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            if not chunks:
                continue
            vecs = await embedder.embed([c.text for c in chunks])
            n = await store.replace_source(name, chunks, vecs)
            total += n
            print(f"  {name}: {n} chunks")
        pruned = await store.prune_missing([name for name, _ in files])
        if pruned:
            print(f"清理已删除文件的孤儿 chunk：{pruned} 条")
        print(f"完成：{len(files)} 个文件，共 {total} chunks")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    loaded = load_files(target)
    if not loaded:
        print(f"{target} 下没有 .md 文件")
    else:
        asyncio.run(ingest(loaded))
