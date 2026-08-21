"""分块器 v1：固定窗口 + 重叠，尽量在段落边界收口。

为什么独立成模块：W2 中期要做「固定 vs 语义 vs 父子文档」对比实验，
接口统一为 chunk(text) -> list[str]，换策略不动调用方。
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    source: str
    index: int
    text: str


def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """固定窗口切分。窗口末尾若在段内，回退到最近的段落/换行边界（最多回退 1/4 窗口）。"""
    if size <= 0 or overlap >= size:
        raise ValueError("需要 size > 0 且 overlap < size")
    out: list[str] = []
    step = size - overlap
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            # 就近对齐段落边界，避免把一句话拦腰切断
            cut = text.rfind("\n\n", i, end)
            if cut == -1:
                cut = text.rfind("\n", i, end)
            if cut != -1 and cut > i + size - size // 4:
                end = cut
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + step)
    return out


def chunk_document(source: str, text: str, *, size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(source=source, index=idx, text=piece)
        for idx, piece in enumerate(_split_fixed(text, size, overlap))
    ]


def _heading_level(line: str) -> int:
    if line.startswith("#"):
        n = len(line) - len(line.lstrip("#"))
        if 1 <= n <= 6 and (len(line) == n or line[n] == " "):
            return n
    return 0


def chunk_markdown(source: str, text: str, *, size: int, overlap: int) -> list[Chunk]:
    """Markdown 结构分块：按标题层级切段，每块前缀「标题路径」保留上下文。

    对 README 类文档，标题就是天然的语义边界——比固定窗口切得更"整"。
    超长小节退回固定窗口二次切分（结构分块的失效场景：无标题的长文）。
    """
    path: list[str] = []  # 当前标题路径，如 ["安装", "macOS"]
    sections: list[tuple[str, list[str]]] = []  # (标题路径串, 行)
    current: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in current):
            sections.append((" > ".join(path), current[:]))
        current.clear()

    for line in text.splitlines():
        level = _heading_level(line)
        if level:
            flush()
            title = line.lstrip("#").strip()
            path[:] = path[: level - 1] + [title]
        else:
            current.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        prefix = f"[{source} > {heading}]\n" if heading else f"[{source}]\n"
        for piece in _split_fixed(body, size, overlap):
            chunks.append(Chunk(source=source, index=len(chunks), text=prefix + piece))
    return chunks


def chunk_by_strategy(
    strategy: str, source: str, text: str, *, size: int, overlap: int
) -> list[Chunk]:
    if strategy == "markdown":
        return chunk_markdown(source, text, size=size, overlap=overlap)
    if strategy == "fixed":
        return chunk_document(source, text, size=size, overlap=overlap)
    raise ValueError(f"未知分块策略: {strategy}（可用: fixed / markdown）")
