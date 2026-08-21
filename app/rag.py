"""RAG 主链路：embed -> 检索 -> 阈值拒答 -> 引用式 prompt -> 生成。"""

from dataclasses import dataclass

from .embeddings import Embedder
from .fusion import rrf_fuse
from .keyword import Bm25Index
from .llm import LlmClient
from .store import Hit, PgStore

_SYSTEM = (
    "你是一个严谨的技术文档问答助手。只依据「资料」回答问题；"
    "每个论断必须标注来源编号，如 [1][2]；"
    "如果资料不足以回答，直接说「资料中没有找到相关信息」，不要编造。"
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict]
    refused: bool
    top_score: float


def build_prompt(question: str, hits: list[Hit]) -> str:
    """引用编号从 1 开始，与返回给前端的 sources 顺序严格一致——溯源的根基。"""
    blocks = [
        f"[{i}] （来源: {h.source} #chunk{h.chunk_index}）\n{h.content}"
        for i, h in enumerate(hits, start=1)
    ]
    return "资料：\n\n" + "\n\n---\n\n".join(blocks) + f"\n\n问题：{question}"


async def answer(
    question: str,
    *,
    embedder: Embedder,
    store: PgStore,
    llm: LlmClient,
    top_k: int,
    min_score: float,
    bm25: Bm25Index | None = None,
) -> RagAnswer:
    qvec = (await embedder.embed([question]))[0]
    # 混合模式两路各取 2k 再融合——RRF 需要足够的候选深度
    vec_hits = await store.search(qvec, top_k * 2 if bm25 else top_k)
    # 拒答阈值只看向量余弦分：BM25 分数无界，不能当置信度（q4 考点）
    top_score = vec_hits[0].score if vec_hits else 0.0
    if bm25 and bm25.size:
        kw_hits = bm25.search(question, top_k * 2)
        hits = rrf_fuse([vec_hits, kw_hits], top_n=top_k)
    else:
        hits = vec_hits[:top_k]

    # 阈值拒答：检索都不相关时不该让 LLM 硬答——省钱且防幻觉
    if not hits or top_score < min_score:
        return RagAnswer(
            answer="资料中没有找到相关信息。",
            sources=[],
            refused=True,
            top_score=top_score,
        )

    text, _in, _out = await llm.complete(_SYSTEM, build_prompt(question, hits))
    sources = [
        {
            "ref": i,
            "source": h.source,
            "chunk_index": h.chunk_index,
            "score": round(h.score, 4),
            "content": h.content,  # 引用原文：前端展示与 faithfulness 评审都需要
        }
        for i, h in enumerate(hits, start=1)
    ]
    return RagAnswer(answer=text, sources=sources, refused=False, top_score=top_score)
