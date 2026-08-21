"""Embedding 客户端：OpenAI 兼容 /embeddings 协议（SiliconFlow / 任何兼容端点通用）。

为什么不默认本地跑 bge-m3：torch + 2GB 权重会把迭代速度拖垮，
API 版先把链路跑通；本地化是 W4 私有化方案里的事。
"""

import hashlib
import math
from typing import Protocol

import httpx

from .config import Settings


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ApiEmbedder:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._s = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        bs = self._s.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            r = await self._client.post(
                f"{self._s.embed_base_url}/embeddings",
                json={"model": self._s.embed_model, "input": batch},
                headers={"Authorization": f"Bearer {self._s.embed_api_key}"},
                timeout=self._s.upstream_timeout,
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out


class FakeEmbedder:
    """测试/离线用：文本哈希 -> 确定性单位向量。相同文本必得相同向量。"""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            raw = [(h[i % 32] - 128) / 128 for i in range(self._dim)]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vecs.append([x / norm for x in raw])
        return vecs


class OllamaEmbedder:
    """本地 ollama 的 bge-m3：免 key、数据不出机——私有化方案里 embedding 本地化的实证。

    注意 trust_env=False：系统代理（HTTP_PROXY 指向 Clash 等）会把 localhost
    请求也劫持成 502——本地模型服务必须绕过代理，这是私有化部署的真实踩坑点。
    """

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = httpx.AsyncClient(trust_env=False)  # 不复用带代理的全局 client
        self._s = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        bs = self._s.embed_batch_size
        for i in range(0, len(texts), bs):
            r = await self._client.post(
                f"{self._s.ollama_base_url}/api/embed",
                json={"model": self._s.ollama_embed_model, "input": texts[i : i + bs]},
                timeout=self._s.upstream_timeout,
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out


def make_embedder(client: httpx.AsyncClient, settings: Settings) -> "Embedder":
    if settings.embed_backend == "ollama":
        return OllamaEmbedder(client, settings)
    return ApiEmbedder(client, settings)
