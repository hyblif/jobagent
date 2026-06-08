import os
import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

QA_QUESTION_RE = re.compile(
    r"^\s*(?:\*{0,2})(?:Q|q|问题?|面试官)\s*\d*\s*[\.:：、)]\s*(.+)$"
)
TRAILING_Q_RE = re.compile(r"^\s*(?:\d+[\.、)]\s*)?(.+[？?])\s*$")
ANSWER_RE = re.compile(r"^\s*(?:\*{0,2})(?:A|a|答案?|回答)\s*[\.:：、)]\s*(.*)")

MAX_CHARS = 700
OVERLAP_CHARS = 80
MIN_CHARS = 15


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _is_question_line(line: str) -> bool:
    return bool(QA_QUESTION_RE.match(line)) or bool(TRAILING_Q_RE.match(line))


def _chunk_section_qa(lines: list[str], meta_base: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[str] = []

    for line in lines:
        if _is_question_line(line) and current:
            text = "\n".join(current).strip()
            if len(text) >= MIN_CHARS:
                chunks.append(Chunk(text=text, metadata={**meta_base, "chunk_type": "qa"}))
            current = [line]
        else:
            current.append(line)

    if current:
        text = "\n".join(current).strip()
        if len(text) >= MIN_CHARS:
            chunks.append(Chunk(text=text, metadata={**meta_base, "chunk_type": "qa"}))

    return chunks


def _chunk_section_prose(body: str, meta_base: dict) -> list[Chunk]:
    paragraphs = re.split(r"\n\s*\n", body)
    chunks: list[Chunk] = []
    current = ""
    overlap = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (overlap + "\n" + para).strip() if overlap else para
        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current and len(current) >= MIN_CHARS:
                chunks.append(Chunk(text=current, metadata={**meta_base, "chunk_type": "prose"}))
                overlap = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else current
            current = para

    if current and len(current) >= MIN_CHARS:
        chunks.append(Chunk(text=current, metadata={**meta_base, "chunk_type": "prose"}))

    return chunks


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    text = text.lstrip("﻿").replace("\r\n", "\n")

    heading_matches = list(HEADING_RE.finditer(text))
    sections: list[tuple[str, str, str]] = []  # (level_marker, title, body)

    if not heading_matches:
        sections.append(("", Path(source).stem, text))
    else:
        if heading_matches[0].start() > 0:
            sections.append(("", Path(source).stem, text[: heading_matches[0].start()]))
        for i, m in enumerate(heading_matches):
            level = m.group(1)
            title = m.group(2).strip()
            body_start = m.end()
            body_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((level, title, body))

    # derive the nearest h1/h2 title for citation display
    h1_title = Path(source).stem
    chunks: list[Chunk] = []

    for level, title, body in sections:
        if level in ("#", "##") and title:
            h1_title = title

        meta_base = {
            "source": source,
            "title": h1_title,
            "heading": title,
        }

        if not body.strip():
            continue

        lines = body.split("\n")
        has_qa = any(_is_question_line(l) for l in lines)

        if has_qa:
            chunks.extend(_chunk_section_qa(lines, meta_base))
        else:
            chunks.extend(_chunk_section_prose(body, meta_base))

    return chunks


def load_markdown_files(data_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    data_path = Path(data_dir)
    for md_file in sorted(data_path.glob("*.md")):
        if md_file.name == "SOURCES.md":
            continue
        text = md_file.read_text(encoding="utf-8")
        file_chunks = chunk_markdown(text, md_file.name)
        chunks.extend(file_chunks)
    return chunks
