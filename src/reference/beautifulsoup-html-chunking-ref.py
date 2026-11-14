#!/usr/bin/env python3
"""
BeautifulSoup reference: parse a messy, "real world" HTML page (nav/header/
footer noise, nested divs, tables, code blocks, multiple heading levels) and
turn it into clean, heading-aware text chunks suitable for a RAG pipeline.

By default this fetches a live page (TARGET_URL) over the network; if the
fetch fails for any reason it falls back to the bundled SAMPLE_HTML so the
script always runs end to end.
Run: python3 src/reference/beautifulsoup-html-chunking-ref.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

# Real page to crawl for this demo -- swap for any URL you want to chunk.
TARGET_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (compatible; rag-chunking-ref/1.0)"

# ---------------------------------------------------------------------------
# A deliberately messy sample page: boilerplate chrome (nav/header/footer),
# scripts/styles, a nested sidebar, multiple heading levels, a table, and a
# code block -- the kind of thing a real docs site actually serves.
# ---------------------------------------------------------------------------
SAMPLE_HTML = """
<html>
<head>
  <title>RAG Pipelines 101</title>
  <style>.sidebar { display: none; }</style>
  <script>console.log("tracking pixel");</script>
</head>
<body>
  <header><div class="logo">ACME Docs</div></header>
  <nav><ul><li><a href="/">Home</a></li><li><a href="/docs">Docs</a></li></ul></nav>

  <main>
    <h1>RAG Pipelines 101</h1>
    <p>Retrieval-Augmented Generation (RAG) combines a retriever with a
    generator so the model can answer using documents it was never
    trained on.</p>

    <div class="content-wrapper">
      <aside class="sidebar"><p>Related: embeddings, vector DBs</p></aside>

      <h2>Chunking Strategies</h2>
      <p>Before you can embed documents, you need to split them into
      chunks that are small enough to embed but large enough to retain
      context.</p>

      <h3>Fixed-size chunking</h3>
      <p>Split text every N characters or tokens, usually with some
      overlap so context isn't lost at the boundary.</p>

      <h3>Semantic chunking</h3>
      <p>Split on natural boundaries like headings, paragraphs, or
      sentence groups instead of a fixed character count.</p>

      <h2>Comparing Chunk Sizes</h2>
      <p>Smaller chunks retrieve more precisely; larger chunks preserve
      more context. A rough comparison:</p>
      <table>
        <tr><th>Chunk Size</th><th>Precision</th><th>Context</th></tr>
        <tr><td>128 tokens</td><td>High</td><td>Low</td></tr>
        <tr><td>512 tokens</td><td>Medium</td><td>Medium</td></tr>
        <tr><td>1024 tokens</td><td>Low</td><td>High</td></tr>
      </table>

      <h2>Example Code</h2>
      <p>A minimal splitter looks like this:</p>
      <pre><code>def split(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]</code></pre>
    </div>
  </main>

  <footer><p>&copy; 2026 ACME. All rights reserved.</p></footer>
</body>
</html>
"""

# Tags that are pure chrome/noise for a RAG knowledge base -- strip them
# before any text extraction happens.
NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside")

# Class-name substrings for boilerplate widgets (navboxes, infoboxes, edit
# links, reference lists) that many real sites -- Wikipedia included -- ship
# alongside the actual article content.
NOISE_CLASS_HINTS = ("navbox", "infobox", "metadata", "noprint", "mw-editsection", "reflist")

# If present, scan only inside this container so sitewide chrome that isn't
# tagged as nav/header/footer is ignored.
CONTENT_CONTAINER_IDS = ("mw-content-text", "main-content", "content")

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


@dataclass
class Section:
    """One heading-bounded slice of the page, with its heading breadcrumb."""

    heading_path: list[str]
    text: str


@dataclass
class Chunk:
    chunk_id: int
    heading_path: str
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def fetch_html(url: str) -> str:
    """Download a page's HTML, raising if the request fails or times out."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def clean_soup(html: str) -> BeautifulSoup:
    """Parse HTML and strip boilerplate chrome that adds no retrieval value."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=lambda c: c and any(hint in c for hint in NOISE_CLASS_HINTS)):
        tag.decompose()
    return soup


def table_to_text(table: Tag) -> str:
    """Flatten an HTML table into pipe-delimited rows so data isn't lost."""
    rows = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all(("th", "td"))]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_sections(soup: BeautifulSoup) -> list[Section]:
    """Walk the content top-to-bottom, grouping text under its nearest headings.

    The heading path (e.g. ["RAG Pipelines 101", "Chunking Strategies",
    "Fixed-size chunking"]) is kept as metadata so a retrieved chunk can show
    its place in the document hierarchy. A level-aware stack is used instead
    of a flat list so this works even when a page's first heading isn't an
    <h1> (e.g. Wikipedia articles start their body at <h2>).
    """
    root = None
    for container_id in CONTENT_CONTAINER_IDS:
        root = soup.find(id=container_id)
        if root:
            break
    root = root or soup.find("main") or soup.find("article") or soup.find("body") or soup

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(heading_path=[title for _, title in stack], text=text))
        buffer.clear()

    for el in root.find_all(list(HEADING_TAGS) + ["p", "table", "pre"], recursive=True):
        if el.name in HEADING_TAGS:
            flush()
            level = int(el.name[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, el.get_text(strip=True)))
        elif el.name == "table":
            buffer.append(table_to_text(el))
        elif el.name == "pre":
            buffer.append(el.get_text())
        else:  # <p>
            buffer.append(el.get_text(strip=True))

    flush()
    return sections


def chunk_sections(sections: list[Section], max_chars: int = 400) -> list[Chunk]:
    """Pack sections into chunks, splitting any section that's too long on its own."""
    chunks: list[Chunk] = []
    for section in sections:
        heading_path = " > ".join(section.heading_path) or "(untitled)"
        text = section.text
        for start in range(0, len(text), max_chars):
            piece = text[start : start + max_chars].strip()
            if piece:
                chunks.append(Chunk(chunk_id=len(chunks), heading_path=heading_path, text=piece))
    return chunks


if __name__ == "__main__":
    try:
        html = fetch_html(TARGET_URL)
        source = TARGET_URL
    except requests.RequestException as exc:
        print(f"Could not fetch {TARGET_URL} ({exc}); falling back to bundled sample HTML.\n")
        html = SAMPLE_HTML
        source = "sample"

    soup = clean_soup(html)
    sections = extract_sections(soup)
    chunks = chunk_sections(sections, max_chars=400)

    print(f"Source: {source}")
    print(f"Extracted {len(sections)} sections -> {len(chunks)} chunks\n")
    for chunk in chunks:
        print(f"[chunk {chunk.chunk_id}] ({chunk.char_count} chars) {chunk.heading_path}")
        print(chunk.text)
        print("-" * 70)
