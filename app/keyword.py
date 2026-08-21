"""BM25 关键词检索通道（jieba 分词 + 内存索引）。

为什么内存 BM25 而不是 ES/Postgres FTS：语料千级，内存索引毫秒级且零运维；
Postgres 中文全文检索还要装 zhparser 扩展。生产规模换 ES——这是个容量决策，不是能力上限。
"""

import jieba
from rank_bm25 import BM25Okapi

from .store import Hit


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


class Bm25Index:
    def __init__(self) -> None:
        self._hits: list[Hit] = []
        self._bm25: BM25Okapi | None = None

    def build(self, rows: list[Hit]) -> None:
        self._hits = rows
        if rows:
            self._bm25 = BM25Okapi([_tokenize(h.content) for h in rows])

    @property
    def size(self) -> int:
        return len(self._hits)

    def search(self, query: str, k: int) -> list[Hit]:
        """返回按 BM25 分数排序的 Hit。分数无界——只用于排名，绝不当置信度用。"""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(
                source=self._hits[i].source,
                chunk_index=self._hits[i].chunk_index,
                content=self._hits[i].content,
                score=float(scores[i]),
            )
            for i in ranked
            if scores[i] > 0
        ]
