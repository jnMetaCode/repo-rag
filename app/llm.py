"""生成端：DeepSeek（OpenAI 兼容），只做 RAG 需要的最小调用。"""

import httpx

from .config import Settings


class LlmClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._s = settings

    async def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """返回 (文本, input_tokens, output_tokens)。"""
        r = await self._client.post(
            f"{self._s.llm_base_url}/chat/completions",
            json={
                "model": self._s.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,  # RAG 生成端要低温：忠于资料而非发挥
            },
            headers={"Authorization": f"Bearer {self._s.llm_api_key}"},
            timeout=self._s.upstream_timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return (
            data["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


class ClaudeCliLlm:
    """本地 claude CLI 当生成端（claude -p 非交互模式）：免 API key。

    与 HTTP 后端同接口（complete）。token 数 CLI 不回报，记 0——
    计费统计在此后端下不可用，属已声明的取舍。
    """

    def __init__(self, model: str = "haiku") -> None:
        self._model = model

    async def complete(self, system: str, user: str) -> tuple[str, int, int]:
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", "--model", self._model,
            "--append-system-prompt", system,
            user,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=180)
        except TimeoutError:
            proc.kill()
            raise RuntimeError("claude CLI 生成超时(180s)") from None
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI 失败: {err.decode(errors='replace')[:200]}")
        return out.decode(errors="replace").strip(), 0, 0


def make_llm(client: httpx.AsyncClient, settings: "Settings"):
    if settings.llm_backend == "claude-cli":
        return ClaudeCliLlm(settings.claude_cli_model)
    return LlmClient(client, settings)
