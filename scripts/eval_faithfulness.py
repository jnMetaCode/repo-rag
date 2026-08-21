"""端到端生成质量评估（免 key：生成与评审都走本地 claude CLI）。

指标（口径对齐 RAGAS，评审器自建）：
- faithfulness：答案中的事实论断被检索资料支持的比例（LLM-judge 逐条判定）
- answer_hit：gold 关键词出现在答案中的比例（机械校验，防 judge 放水）
- refusal_acc：6 道应拒题的正确拒答率（检索闸门拒 or LLM 层拒都算对）

用法：uv run python scripts/eval_faithfulness.py [--limit N]
"""

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.embeddings import make_embedder  # noqa: E402
from app.llm import ClaudeCliLlm  # noqa: E402
from app.rag import answer  # noqa: E402
from app.store import PgStore  # noqa: E402

SEM = asyncio.Semaphore(4)  # claude CLI 并发上限
REFUSE_PAT = re.compile(r"没有找到|资料不足|没有相关|无法回答|不涉及|没有提及|未提及|无相关")

JUDGE_SYSTEM = (
    "你是严格的事实核查员。给你一份「资料」和一段「答案」。"
    "把答案拆成独立的事实性论断（忽略客套话/格式），逐条判断是否被资料支持。"
    '只输出 JSON：{"total": 论断数, "supported": 被支持数, "unsupported": ["未被支持的论断"]}'
)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def run_item(g: dict, embedder, store, gen_llm, judge_llm) -> dict:
    async with SEM:
        result = await answer(
            g["question"], embedder=embedder, store=store, llm=gen_llm,
            top_k=settings.top_k, min_score=settings.min_score, bm25=None,
        )
    row: dict = {"id": g["id"], "type": g["type"], "refused_gate": result.refused,
                 "top_score": result.top_score}
    if g["type"] == "refusal":
        row["refused_ok"] = result.refused or bool(REFUSE_PAT.search(result.answer))
        row["answer_head"] = result.answer[:80]
        return row
    # 检索题：关键词命中（机械） + faithfulness（judge）
    kws = g.get("keywords") or []
    row["kw_hit"] = sum(1 for k in kws if k in result.answer)
    row["kw_total"] = len(kws)
    if result.refused:
        row["faithfulness"] = 0.0
        row["note"] = "误拒"
        return row
    ctx = "\n\n---\n\n".join(f"[{src['ref']}] {src['content']}" for src in result.sources)
    async with SEM:
        verdict_text, _, _ = await judge_llm.complete(
            JUDGE_SYSTEM, f"资料：\n{ctx}\n\n答案：\n{result.answer}"
        )
    v = parse_json(verdict_text)
    if v and v.get("total"):
        row["faithfulness"] = round(v["supported"] / v["total"], 3)
        row["unsupported"] = v.get("unsupported", [])[:3]
    else:
        row["faithfulness"] = None
        row["note"] = "judge 输出不可解析"
    return row


async def main() -> None:
    gold = [json.loads(x) for x in (ROOT / "eval/gold.jsonl").read_text().splitlines() if x.strip()]
    if "--limit" in sys.argv:
        n = int(sys.argv[sys.argv.index("--limit") + 1])
        gold = gold[:n]
    store = PgStore(settings.db_dsn, settings.embed_dim)
    gen_llm = ClaudeCliLlm(settings.claude_cli_model)
    judge_llm = ClaudeCliLlm(settings.claude_cli_model)
    async with httpx.AsyncClient() as client:
        embedder = make_embedder(client, settings)
        rows = await asyncio.gather(
            *[run_item(g, embedder, store, gen_llm, judge_llm) for g in gold]
        )
    ret = [r for r in rows if r["type"] == "retrieval"]
    ref = [r for r in rows if r["type"] == "refusal"]
    faiths = [r["faithfulness"] for r in ret if r.get("faithfulness") is not None]
    kw = sum(r["kw_hit"] for r in ret), sum(r["kw_total"] for r in ret)
    print(f"\n== 生成质量 (n={len(ret)}) ==")
    print(f"  faithfulness 均值: {sum(faiths)/len(faiths):.3f}"
          f"  （满分 1.0 条目: {sum(1 for f in faiths if f >= 0.999)}/{len(faiths)}）")
    print(f"  关键词命中: {kw[0]}/{kw[1]} = {kw[0]/kw[1]:.1%}")
    low = [r for r in ret if (r.get("faithfulness") or 1) < 0.8]
    for r in low:
        print(f"  ⚠ #{r['id']} faith={r['faithfulness']} 未支持论断: {r.get('unsupported')}")
    print(f"\n== 拒答 (n={len(ref)}) ==")
    ok = sum(1 for r in ref if r["refused_ok"])
    print(f"  正确拒答: {ok}/{len(ref)}"
          f"  （闸门拒: {sum(1 for r in ref if r['refused_gate'])} · LLM层拒: "
          f"{sum(1 for r in ref if r['refused_ok'] and not r['refused_gate'])}）")
    for r in ref:
        mark = "✓" if r["refused_ok"] else "✗ 未拒答！"
        gate = "闸门" if r["refused_gate"] else f"LLM层(top1={r['top_score']:.2f})"
        print(f"  {mark} #{r['id']} [{gate}] {r.get('answer_head', '')[:50]}")
    (ROOT / "eval" / "faithfulness-latest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n明细已存 eval/faithfulness-latest.json")


asyncio.run(main())
