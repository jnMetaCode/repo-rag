"""检索评估：hit@k / MRR，按通道分别测。

- bm25 通道：零依赖，随时可跑（本脚本默认）
- vector / hybrid 通道：--vector 开启，需要 DB 已 ingest + RAG_EMBED_API_KEY

用法：uv run python scripts/eval_retrieval.py [--vector]
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.chunker import chunk_by_strategy  # noqa: E402
from app.config import settings  # noqa: E402
from app.keyword import Bm25Index  # noqa: E402
from app.store import Hit  # noqa: E402

K = 5


def load_gold() -> list[dict]:
    lines = (ROOT / "eval" / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def load_chunks() -> list[Hit]:
    hits: list[Hit] = []
    for f in sorted((ROOT / "data").glob("*.md")):
        text = f.read_text(encoding="utf-8", errors="replace")
        for c in chunk_by_strategy(
            settings.chunk_strategy, f.name, text,
            size=settings.chunk_size, overlap=settings.chunk_overlap,
        ):
            hits.append(Hit(source=c.source, chunk_index=c.index, content=c.text, score=0.0))
    return hits


def score_channel(name: str, gold: list[dict], search) -> None:
    items = [g for g in gold if g["type"] == "retrieval"]
    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    misses: list[tuple[int, str, list[str]]] = []
    for g in items:
        results = search(g["question"])
        srcs = [h.source for h in results[:K]]
        rank = next((i + 1 for i, s in enumerate(srcs) if s in g["expected_sources"]), None)
        if rank:
            mrr += 1 / rank
            hit1 += rank <= 1
            hit3 += rank <= 3
            hit5 += 1
        else:
            misses.append((g["id"], g["question"], srcs))
    n = len(items)
    print(f"\n== {name}  (n={n}, top{K}) ==")
    print(f"  hit@1 {hit1/n:.2%}   hit@3 {hit3/n:.2%}   hit@5 {hit5/n:.2%}   MRR {mrr/n:.3f}")
    for gid, q, srcs in misses:
        print(f"  ✗ #{gid} {q}\n      召回: {srcs}")


async def main() -> None:
    gold = load_gold()
    chunks = load_chunks()
    print(f"语料: {len(chunks)} chunks（策略={settings.chunk_strategy}）")

    bm25 = Bm25Index()
    bm25.build(chunks)
    score_channel("BM25", gold, lambda q: bm25.search(q, K))

    if "--vector" in sys.argv:
        import httpx

        from app.embeddings import make_embedder
        from app.fusion import rrf_fuse
        from app.store import PgStore

        store = PgStore(settings.db_dsn, settings.embed_dim)
        async with httpx.AsyncClient() as client:
            embedder = make_embedder(client, settings)

            async def vec_search(q: str) -> list[Hit]:
                v = (await embedder.embed([q]))[0]
                return await store.search(v, K)

            async def hybrid_search(q: str) -> list[Hit]:
                v = (await embedder.embed([q]))[0]
                return rrf_fuse(
                    [await store.search(v, K * 2), bm25.search(q, K * 2)], top_n=K
                )

            # 同步壳跑异步通道
            vec_map = {g["question"]: await vec_search(g["question"]) for g in gold}
            hyb_map = {g["question"]: await hybrid_search(g["question"]) for g in gold}
        await store.close()
        score_channel("Vector", gold, vec_map.__getitem__)
        score_channel("Hybrid(RRF)", gold, hyb_map.__getitem__)


if __name__ == "__main__":
    asyncio.run(main())
