"""分块器：边界条件是这类代码 bug 的全部来源。"""

import pytest

from app.chunker import chunk_document


def test_short_text_single_chunk():
    chunks = chunk_document("a.md", "短文本", size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "短文本"
    assert chunks[0].index == 0


def test_overlap_and_coverage():
    text = "".join(f"第{i}段落的内容在这里。\n\n" for i in range(60))
    chunks = chunk_document("a.md", text, size=200, overlap=40)
    assert len(chunks) > 1
    # 覆盖性：每个原始段落都出现在至少一个 chunk 里
    for i in range(60):
        assert any(f"第{i}段落" in c.text for c in chunks), f"第{i}段落丢失"
    # index 连续
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_paragraph_boundary_alignment():
    text = ("A" * 180) + "\n\n" + ("B" * 180)
    chunks = chunk_document("a.md", text, size=200, overlap=20)
    # 窗口尾部落在 B 段内，应回退到段落边界：首块只含 A
    assert chunks[0].text == "A" * 180


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_document("a.md", "x", size=100, overlap=100)


def test_empty_text():
    assert chunk_document("a.md", "   \n  ", size=100, overlap=10) == []
