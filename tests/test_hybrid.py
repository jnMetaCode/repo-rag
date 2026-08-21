"""D8–11 新增件：markdown 分块 / BM25 通道 / RRF 融合 / 混合链路。"""

import httpx
import respx

from app.chunker import chunk_by_strategy, chunk_markdown
from app.config import settings
from app.embeddings import FakeEmbedder
from app.fusion import rrf_fuse
from app.keyword import Bm25Index
from app.llm import LlmClient
from app.rag import answer
from app.store import Hit

LLM_URL = "https://api.deepseek.com/chat/completions"


# ---------- markdown 分块 ----------

MD = """# 项目简介
这是一个多智能体编排引擎。

## 安装
### macOS
用 brew 安装即可。

## 使用
运行 orchestrator 命令启动。
"""


def test_markdown_chunk_heading_path():
    chunks = chunk_markdown("orc.md", MD, size=512, overlap=64)
    texts = [c.text for c in chunks]
    # 标题路径作为上下文前缀
    assert any("[orc.md > 项目简介]" in t for t in texts)
    assert any("[orc.md > 项目简介 > 安装 > macOS]" in t and "brew" in t for t in texts)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_markdown_oversized_section_falls_back_to_window():
    long_md = "# 长节\n" + "内容句子。" * 400  # 2000 字符，超过 512 窗口
    chunks = chunk_markdown("a.md", long_md, size=512, overlap=64)
    assert len(chunks) > 1
    assert all(t.text.startswith("[a.md > 长节]") for t in chunks)


def test_strategy_dispatch():
    assert chunk_by_strategy("fixed", "a.md", "文本", size=100, overlap=10)
    assert chunk_by_strategy("markdown", "a.md", "# t\n文本", size=100, overlap=10)
    try:
        chunk_by_strategy("semantic", "a.md", "x", size=100, overlap=10)
        raise AssertionError("应当抛 ValueError")
    except ValueError:
        pass


# ---------- BM25 ----------


def _h(i, content):
    return Hit(source=f"f{i}.md", chunk_index=i, content=content, score=0.0)


def test_bm25_chinese_exact_term_first():
    idx = Bm25Index()
    idx.build(
        [
            _h(0, "agency-orchestrator 是一个多智能体编排引擎，支持 YAML 工作流"),
            _h(1, "今天天气很好，适合出去散步"),
            _h(2, "本项目提供视频提示词生成能力"),
        ]
    )
    hits = idx.search("编排引擎 YAML", k=2)
    assert hits and hits[0].chunk_index == 0


def test_bm25_empty_index():
    assert Bm25Index().search("任何词", k=5) == []


# ---------- RRF ----------


def test_rrf_doc_in_both_lists_wins():
    a, b, c = _h(1, "A"), _h(2, "B"), _h(3, "C")
    # A 在两路都排第 2；B、C 各只在一路排第 1
    fused = rrf_fuse([[b, a], [c, a]], k=60, top_n=3)
    # 1/62+1/62 > 1/61 → A 融合分最高
    assert (fused[0].source, fused[0].chunk_index) == ("f1.md", 1)
    assert len(fused) == 3


def test_rrf_score_math():
    a = _h(1, "A")
    fused = rrf_fuse([[a], [a]], k=60, top_n=1)
    assert abs(fused[0].score - 2 / 61) < 1e-6  # 融合分展示时舍入到 6 位


# ---------- 混合链路 ----------


class FakeStore:
    def __init__(self, hits):
        self._hits = hits

    async def search(self, query_vec, k):
        return self._hits[:k]


@respx.mock
async def test_hybrid_answer_merges_channels():
    respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "见 [1]。"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    vec_hits = [
        Hit(source="v.md", chunk_index=0, content="向量召回的内容", score=0.8),
        Hit(source="both.md", chunk_index=9, content="两路共同召回 编排引擎", score=0.7),
    ]
    idx = Bm25Index()
    # 注意：语料必须有"不含查询词"的文档，否则查询词 df=N → IDF≤0 → BM25 全零。
    # （词在所有文档都出现 = 零区分度，这是 BM25 的定义行为，不是 bug）
    idx.build(
        [
            Hit(source="both.md", chunk_index=9, content="两路共同召回 编排引擎", score=0.0),
            Hit(source="k.md", chunk_index=1, content="关键词通道 独有 编排引擎 文档", score=0.0),
            Hit(source="x1.md", chunk_index=2, content="今天天气很好适合散步", score=0.0),
            Hit(source="x2.md", chunk_index=3, content="红烧肉的做法需要五花肉", score=0.0),
            Hit(source="x3.md", chunk_index=4, content="前端页面使用组件化开发", score=0.0),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await answer(
            "编排引擎",
            embedder=FakeEmbedder(dim=8),
            store=FakeStore(vec_hits),
            llm=LlmClient(client, settings),
            top_k=3,
            min_score=0.35,
            bm25=idx,
        )
    assert result.refused is False
    srcs = {s["source"] for s in result.sources}
    # 双路命中的 both.md 必在；BM25 独有的 k.md 也应进融合结果
    assert "both.md" in srcs and "k.md" in srcs
    # 双路命中者排第一（RRF 性质）
    assert result.sources[0]["source"] == "both.md"


async def test_hybrid_refusal_still_uses_vector_score():
    """BM25 命中很多但向量 top1 低于阈值 → 仍拒答（BM25 分数无界，不能当置信度）。"""
    idx = Bm25Index()
    idx.build([_h(0, "编排引擎 相关内容")])
    result = await answer(
        "编排引擎",
        embedder=FakeEmbedder(dim=8),
        store=FakeStore([Hit(source="v.md", chunk_index=0, content="弱相关", score=0.1)]),
        llm=LlmClient(httpx.AsyncClient(), settings),
        top_k=3,
        min_score=0.35,
        bm25=idx,
    )
    assert result.refused is True
