# repo-rag

把开源仓库文档做成可问答的中文知识库：**bge-m3 向量检索 · 引用溯源 · 阈值拒答 · pgvector**。

> Built in public：一个用「评估驱动」方法论从零搭起来的 RAG——每个技术决策（分块/检索通道/拒答阈值）都由 30 条黄金测试集的实测数据推导，而不是照抄最佳实践。

## 跑起来（runbook）

```bash
docker compose up -d db          # pgvector
uv sync
cp .env.example .env             # 填 RAG_EMBED_API_KEY（SiliconFlow）+ RAG_LLM_API_KEY（DeepSeek）
uv run python scripts/fetch_corpus.py jnMetaCode 20   # 抓语料到 data/
uv run python -m app.ingest data                      # 分块+向量化+入库
uv run uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question": "agency-orchestrator 是做什么的？"}' | jq
curl -s localhost:8000/v1/stats
```

测试（无需 key、无需数据库——集成测试没库时自动跳过）：

```bash
uv run pytest -q && uv run ruff check .
```

## 链路（混合检索版）

```
question ──┬─ embed(bge-m3) ─▶ pgvector cosine top-2k ─┐
           │                     │                      ├─▶ RRF 融合 ─▶ top-k
           └─ jieba 分词 ──────▶ BM25 内存索引 top-2k ──┘
                                 │
                 向量 top1 < 0.35 ─▶ 直接拒答（不调 LLM）
                                 ▼
            [n] 引用编号 prompt ─▶ DeepSeek(temp=0.2) ─▶ answer + sources[]
```

入库后调 `POST /v1/reindex` 重建 BM25 索引（免重启）；`GET /v1/stats` 看两路状态。

## 实测结果（2026-08-21 · 30 条 gold 集 · 全本地免 key 栈）

| 通道 | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| BM25（jieba） | 54.2% | 75.0% | 91.7% | 0.678 |
| **向量（bge-m3 via ollama）** | **95.8%** | **100%** | **100%** | **0.979** |
| 混合 RRF | 83.3% | 100% | 100% | 0.910 |

三个数据驱动的结论：
① BM25 的 miss 全是跨语言（中文问印尼语文档）→ 这就是 bge-m3 选型的实证依据；
② 本语料上混合反而拖低 hit@1 → 默认改纯向量（混合留给含错误码/精确 ID 的语料）；
③ 拒答阈值校准：应答题 ≥0.615，远域应拒 ≤0.479 → 阈值 0.50；近域无答案题（0.56–0.65）
  检索分无法识别 → **两层拒答**：检索闸门拦远域，LLM 层（低温+明示拒答指令）兜近域，实测兜住。

**端到端生成质量（claude CLI 当 judge，口径对齐 RAGAS）**：
faithfulness 均值 **0.981**（19/24 满分）· 关键词命中 88.9% · **拒答 6/6 全对**——
其中 3 道远域被检索闸门拦截、3 道近域（top1 0.56–0.65）被 LLM 层兜住，
与阈值校准时的分数分布预测完全吻合：两层拒答架构从设计→校准→实测三步闭环。

免 key 本地栈：embedding = ollama bge-m3（16s 入库 502 chunks），生成 = 本地 claude CLI。
踩坑记录：系统代理（Clash）会把 localhost 请求劫持成 502——本地模型客户端必须 trust_env=False。

## 设计决策（面试可讲）

1. **拒答先于生成**：top1 相似度低于阈值直接返回"资料不足"——幻觉抑制的第一道闸，还省 token。阈值 0.35 是起点，D12 用测试集校准。
2. **引用编号 = 溯源契约**：prompt 里的 [n] 和响应 sources[].ref 由同一段代码生成，保证可核对（`test_answer_with_sources` 锁死这个一致性）。
3. **生成端 temperature=0.2**：RAG 要忠于资料，不要创造力。
4. **embedding 走 API 不本地跑**：torch+2GB 权重拖慢迭代；本地化留给 W4 私有化方案。
5. **入库只走 CLI**：无鉴权的 HTTP 写接口 = 事故。
6. **分块策略可切换**：`fixed`（固定窗口）/ `markdown`（标题即语义边界，块前缀标题路径保上下文）——README 类文档默认 markdown；无标题长文是它的失效场景，自动退回窗口切分。
7. **RRF 而非加权融合**：BM25 分数无界、余弦有界，分布不可比；RRF 只用排名，免调参（k=60，Cormack 2009）。**拒答阈值仍只看向量余弦分——BM25 分数不能当置信度。**
8. **BM25 用内存索引而非 ES**：千级语料毫秒级、零运维；Postgres 中文 FTS 还要装 zhparser。生产规模换 ES 是容量决策，不是能力上限。

## Java 对照

psycopg AsyncConnection ≈ JDBC + HikariCP（v1 每操作一连，连接池是 D10 优化项——先跑通再优化，这本身就是决策）。

## 已知边界（v1 刻意不做）

多轮对话、增量更新、rerank（bge-reranker，下一步）、RAGAS（D12-13）、鉴权、前端。
