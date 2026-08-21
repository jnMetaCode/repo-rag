"""配置。环境变量前缀 RAG_，.env 亦可。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    db_dsn: str = "postgresql://rag:rag@localhost:5433/rag"

    # 后端选择：本地免 key 栈（ollama + claude CLI）为默认；有 API key 时切 api/openai
    embed_backend: str = "ollama"      # ollama / api
    llm_backend: str = "claude-cli"    # claude-cli / api
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "bge-m3"
    claude_cli_model: str = "haiku"    # RAG 生成端要快和稳，不需要最大杯

    # Embedding API 备选：OpenAI 兼容 /embeddings 端点（SiliconFlow 的 bge-m3，1024 维）。
    embed_api_key: str = ""
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 64

    # 生成端：DeepSeek（OpenAI 兼容）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    upstream_timeout: float = 60.0

    chunk_size: int = 512      # 字符窗口
    chunk_overlap: int = 64
    chunk_strategy: str = "markdown"  # fixed / markdown（README 类文档标题即语义边界）
    # 实测（gold 24 题）：纯向量 hit@1 95.8%/MRR 0.979，混合 RRF hit@1 83.3%/MRR 0.910——
    # 本语料向量占优，BM25 反而拖低 top1，故默认纯向量；含大量错误码/精确 ID 的语料再开混合。
    hybrid: bool = False
    top_k: int = 5
    # 拒答阈值：用 30 条 gold 集实测校准（2026-08-21）——
    # 应答题 top1 ∈ [0.615, 0.781]，远域应拒 ∈ [0.397, 0.479] → 0.50 干净分界；
    # 近域无答案题(0.56–0.65)检索分无法识别，由 LLM 层兜底（两层拒答架构）。
    min_score: float = 0.50


settings = Settings()
