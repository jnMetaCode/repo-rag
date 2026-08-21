"""RRF（Reciprocal Rank Fusion）：score = Σ 1/(k + rank)。

为什么不用加权平均：BM25 分数无界、余弦相似度有界 [−1,1]，分布不可比，
加权需要调参且换语料就失效。RRF 只看排名——免调参、对异常分数鲁棒。
k=60 是原论文（Cormack et al. 2009）推荐值。
"""

from .store import Hit


def rrf_fuse(rankings: list[list[Hit]], *, k: int = 60, top_n: int = 5) -> list[Hit]:
    scores: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.source, hit.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # 保留向量通道的原始余弦分（拒答阈值仍要用它）；同 key 取分高的展示
            if key not in best or hit.score > best[key].score:
                best[key] = hit
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_n]
    return [
        Hit(
            source=key[0],
            chunk_index=key[1],
            content=best[key].content,
            score=round(scores[key], 6),  # 展示 RRF 分
        )
        for key in ordered
    ]
