"""RAG 主链路：prompt 构造 / 阈值拒答 / 引用一致性（store 和 LLM 全 fake/mock）。"""

import httpx
import respx

from app.config import settings
from app.embeddings import FakeEmbedder
from app.llm import LlmClient
from app.rag import RagAnswer, answer, build_prompt
from app.store import Hit

LLM_URL = "https://api.deepseek.com/chat/completions"


class FakeStore:
    def __init__(self, hits):
        self._hits = hits

    async def search(self, query_vec, k):
        return self._hits[:k]


def _hit(i, score):
    return Hit(source=f"repo{i}.md", chunk_index=i, content=f"内容{i}", score=score)


def test_build_prompt_citation_order():
    hits = [_hit(1, 0.9), _hit(2, 0.8)]
    p = build_prompt("问题？", hits)
    assert p.index("[1]") < p.index("[2]")
    assert "repo1.md" in p and "内容2" in p and p.endswith("问题：问题？")


async def test_refusal_below_threshold():
    """top1 分数低于阈值：不调 LLM（respx 未挂 mock 也不会炸），直接拒答。"""
    result = await answer(
        "无关问题",
        embedder=FakeEmbedder(dim=8),
        store=FakeStore([_hit(1, 0.10)]),
        llm=LlmClient(httpx.AsyncClient(), settings),
        top_k=3,
        min_score=0.35,
    )
    assert isinstance(result, RagAnswer)
    assert result.refused is True
    assert result.sources == []
    assert "没有找到" in result.answer


async def test_empty_store_refuses():
    result = await answer(
        "问题",
        embedder=FakeEmbedder(dim=8),
        store=FakeStore([]),
        llm=LlmClient(httpx.AsyncClient(), settings),
        top_k=3,
        min_score=0.35,
    )
    assert result.refused is True


@respx.mock
async def test_answer_with_sources():
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "根据 [1]，答案是 X。"}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await answer(
            "X 是什么？",
            embedder=FakeEmbedder(dim=8),
            store=FakeStore([_hit(1, 0.82), _hit(2, 0.71)]),
            llm=LlmClient(client, settings),
            top_k=5,
            min_score=0.35,
        )
    assert result.refused is False
    assert "[1]" in result.answer
    # sources 的 ref 编号必须与 prompt 中 [n] 完全一致
    assert [s["ref"] for s in result.sources] == [1, 2]
    assert result.sources[0]["source"] == "repo1.md"
    # 生成端低温：温度参数确实传了 0.2
    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent["temperature"] == 0.2


async def test_fake_embedder_deterministic():
    e = FakeEmbedder(dim=16)
    v1 = await e.embed(["同一段文本"])
    v2 = await e.embed(["同一段文本"])
    assert v1 == v2
    norm = sum(x * x for x in v1[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6
