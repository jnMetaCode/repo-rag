# repo-rag

把开源仓库文档做成可问答的中文知识库：**bge-m3 向量检索 · 引用溯源 · 阈值拒答 · pgvector**。

> Built in public：一个用「评估驱动」方法论搭起来的 RAG——每个技术决策（分块 / 检索通道 / 拒答阈值）
> 都由 30 条黄金测试集的实测数据推导，而不是照抄最佳实践。**全链路零 API key**，本地跑得动。

| 检索 hit@1 | hit@5 | MRR | faithfulness | 拒答准确 |
|---|---|---|---|---|
| **95.8%** | **100%** | **0.979** | **0.981** | **6/6** |

<sub>30 条自建黄金集实测（24 检索 + 6 拒答）· 口径与复现命令见下方「实测结果」</sub>

**三个和最佳实践相反的结论**：① BM25 基线 hit@5 91.7% 看着够用，但 **miss 全是跨语言**——这才是换 bge-m3 的理由；
② 混合检索 RRF 在这个语料上**反而把 hit@1 拉低到 83.3%**，所以默认关掉了；
③ 拒答阈值不是拍的，是跑分数分布校准出来的（0.50），而且有一类问题**检索分原理上识别不了**，必须靠第二层兜。

## 跑起来（runbook）

**默认路径：零 API key**（embedding 走本地 ollama，生成走本地 claude CLI）

```bash
ollama pull bge-m3               # 本地 embedding，1.2G
docker compose up -d db          # pgvector
uv sync
uv run python scripts/fetch_corpus.py <你的 GitHub 用户名> 20   # 抓语料到 data/
uv run python -m app.ingest data                              # 分块+向量化+入库（502 chunks 约 16s）
uv run uvicorn app.main:app --port 8001                       # 打开 http://localhost:8001
```

想换成云端 API：`cp .env.example .env`，填 `RAG_EMBED_API_KEY`（任何 OpenAI 兼容 embeddings 端点）与
`RAG_LLM_API_KEY`（DeepSeek），并把 `RAG_EMBED_BACKEND=api`、`RAG_LLM_BACKEND=api`。

> ⚠️ 本机若有系统代理（Clash 等），它会劫持到 localhost 的请求并返回 502——
> 代码里 ollama 客户端已设 `trust_env=False` 绕开，这是私有化部署的典型踩坑点。

```bash
curl -s localhost:8001/v1/query -H 'content-type: application/json' \
  -d '{"question": "agency-orchestrator 是做什么的？"}' | jq
curl -s localhost:8001/v1/stats
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

---

### 关于这一组项目

这是三套**评估驱动**的 AI 应用系统，同期开源，可以单独用也可以对照看：

| | 做什么 | 关键实测 |
|---|---|---|
| [repo-rag](https://github.com/jnMetaCode/repo-rag) | 中文知识库 RAG：结构分块 + 两层拒答 + 引用溯源 | hit@1 95.8% · faithfulness 0.981 |
| [orchestrator-lg](https://github.com/jnMetaCode/orchestrator-lg) | 自研 DAG 引擎迁到 LangGraph：checkpoint + 可持久化审批中断 | 7/7 测试 · YAML 零改动兼容 |
| [llm-gateway](https://github.com/jnMetaCode/llm-gateway) | 多模型网关：SSE 取消链 + 三态熔断 + token 计费 | 10/10 测试 · Docker |

共同的方法论：**先建评估集，再写优化**——每个技术决策都由实测数据推导，包括那些「该做但做了反而更差」的决策。

### 关于作者

[@jnMetaCode](https://github.com/jnMetaCode) · 11 年 IT、8 年技术团队管理 · 公众号 **AI不止语**
其他开源：[agency-agents-zh](https://github.com/jnMetaCode/agency-agents-zh)（19.8k★，267 个 AI 专家角色 × 18 类工具链）·
[superpowers-zh](https://github.com/jnMetaCode/superpowers-zh)（7.8k★）· [agency-orchestrator](https://github.com/jnMetaCode/agency-orchestrator)（2.1k★，本项目的上游）
