"""Tests for src/rag/ingest.py — pure logic, no models."""
import pytest

from src.rag.ingest import MIN_CHARS, chunk_markdown, load_markdown_files


QA_MD = """\
# Python 八股

## 并发

Q1: 什么是 GIL？
A: GIL 是全局解释器锁，保证同一时刻只有一个线程执行字节码。

Q2: 协程和线程的区别？
A: 协程是用户态调度，线程是内核态调度。
"""

PROSE_MD = """\
# 操作系统

## 内存管理

虚拟内存让每个进程拥有独立的逻辑地址空间，通过页表映射到物理内存。
分页将地址空间划分为固定大小的页（通常 4KB）。

TLB 缓存热点映射，加速地址转换。
"""

TRAILING_Q_MD = """\
# OS

## 基础

什么是死锁？
死锁是两个进程互相等待对方释放资源。

进程和线程的区别？
进程是资源分配的最小单位，线程是 CPU 调度的最小单位。
"""


def test_qa_chunks_split_by_question():
    chunks = chunk_markdown(QA_MD, "python.md")
    qa_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "qa"]
    assert len(qa_chunks) == 2
    assert "GIL" in qa_chunks[0].text
    assert "协程" in qa_chunks[1].text


def test_qa_chunk_metadata():
    chunks = chunk_markdown(QA_MD, "python.md")
    for c in chunks:
        assert "source" in c.metadata
        assert c.metadata["source"] == "python.md"
        assert "title" in c.metadata
        assert "heading" in c.metadata


def test_prose_chunks_created():
    chunks = chunk_markdown(PROSE_MD, "os.md")
    prose_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "prose"]
    assert len(prose_chunks) >= 1


def test_trailing_question_detection():
    chunks = chunk_markdown(TRAILING_Q_MD, "os.md")
    qa_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "qa"]
    assert len(qa_chunks) >= 2


def test_short_chunks_dropped():
    short_md = "# 标题\n\n## 节\n\nok\n"
    chunks = chunk_markdown(short_md, "test.md")
    for c in chunks:
        assert len(c.text.strip()) >= MIN_CHARS


def test_crlf_normalization():
    md_crlf = "# 标题\r\n\r\n## 节\r\n\r\nQ1: 问题？\r\nA: 答案内容在这里，足够长。\r\n"
    chunks = chunk_markdown(md_crlf, "test.md")
    assert len(chunks) >= 1
    for c in chunks:
        assert "\r" not in c.text


def test_bom_stripped():
    md_bom = "﻿# 标题\n\n## 节\n\nQ1: 问题？\nA: 答案足够长的内容在这里。\n"
    chunks = chunk_markdown(md_bom, "test.md")
    assert len(chunks) >= 1
    assert not chunks[0].text.startswith("﻿")


def test_load_markdown_files(tmp_path):
    (tmp_path / "test1.md").write_text(QA_MD, encoding="utf-8")
    (tmp_path / "test2.md").write_text(PROSE_MD, encoding="utf-8")
    (tmp_path / "SOURCES.md").write_text("ignore me", encoding="utf-8")
    chunks = load_markdown_files(str(tmp_path))
    sources = {c.metadata["source"] for c in chunks}
    assert "test1.md" in sources
    assert "test2.md" in sources
    assert "SOURCES.md" not in sources
