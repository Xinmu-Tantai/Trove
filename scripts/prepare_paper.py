#!/usr/bin/env python3
"""Prepare a paper PDF for the Trove extraction workflow.

The script performs reproducible mechanical steps only: it stores a source PDF,
extracts page-delimited text, and optionally renders selected PDF pages for
figure and table review. It does not generate the final article note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def parse_pages(value: str) -> list[int]:
    try:
        pages = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("页码应为逗号分隔的正整数") from error
    if not pages or pages[0] < 1:
        raise argparse.ArgumentTypeError("页码应为正整数")
    return pages


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(title: str) -> str:
    """Keep a readable title while excluding path and cross-platform reserved characters."""
    filename = re.sub(r'[<>:"/\\|?*]+', " - ", title)
    filename = " ".join(filename.split()).strip(". ")
    if not filename:
        raise ValueError("论文标题不能生成有效文件名")
    return filename


def metadata_title(pdf_path: Path) -> str | None:
    title = (PdfReader(pdf_path).metadata or {}).get("/Title")
    return str(title).strip() or None


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Trove paper preparer"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def extract_text(pdf_path: Path, destination: Path) -> int:
    reader = PdfReader(pdf_path)
    with destination.open("w", encoding="utf-8") as file:
        for number, page in enumerate(reader.pages, start=1):
            file.write(f"\n\n===== PAGE {number} =====\n\n")
            file.write(page.extract_text(extraction_mode="layout") or "")
    return len(reader.pages)


def render_pages(pdf_path: Path, pages: list[int], output_dir: Path, total_pages: int) -> None:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "渲染页面需要 PyMuPDF：python3 -m pip install PyMuPDF"
        ) from error

    document = fitz.open(pdf_path)
    for page_number in pages:
        if page_number > total_pages:
            raise ValueError(f"页码 {page_number} 超出 PDF 的 {total_pages} 页")
        pixmap = document[page_number - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(output_dir / f"page-{page_number}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="为 Trove 论文提取准备 PDF、文本和页面渲染。")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path, help="已有 PDF 的路径")
    source.add_argument("--url", help="待下载的 PDF URL")
    parser.add_argument("--article-id", required=True, help="文章目录名，如 openvla")
    parser.add_argument(
        "--title",
        help="论文标题；省略时使用 PDF 元数据中的 Title。元数据为空时必须提供。",
    )
    parser.add_argument(
        "--render-pages",
        type=parse_pages,
        help="需渲染核对的页码，例如 1,4,7,10",
    )
    args = parser.parse_args()

    article_id = args.article_id.strip()
    if not article_id or "/" in article_id or "\\" in article_id or article_id in {".", ".."}:
        parser.error("article-id 必须是单个有效目录名")

    sources_dir = ROOT / "sources"
    work_dir = ROOT / "work" / article_id
    sources_dir.mkdir(exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        title = args.title
        if not title:
            parser.error("使用 --url 时请同时提供 --title，以论文标题命名 PDF")
        pdf_path = sources_dir / f"{safe_filename(title)}.pdf"
        download(args.url, pdf_path)
    else:
        source_pdf = args.pdf.resolve()
        if not source_pdf.is_file():
            parser.error(f"未找到 PDF：{source_pdf}")
        title = args.title or metadata_title(source_pdf)
        if not title:
            parser.error("PDF 元数据没有 Title，请通过 --title 提供论文标题")
        pdf_path = sources_dir / f"{safe_filename(title)}.pdf"
        if source_pdf != pdf_path.resolve():
            if pdf_path.exists():
                if sha256(source_pdf) != sha256(pdf_path):
                    parser.error(f"目标文件已存在且内容不同：{pdf_path}")
                if source_pdf.parent == sources_dir.resolve():
                    source_pdf.unlink()
            elif source_pdf.parent == sources_dir.resolve():
                source_pdf.replace(pdf_path)
            else:
                shutil.copy2(source_pdf, pdf_path)

    text_path = work_dir / "paper.txt"
    page_count = extract_text(pdf_path, text_path)

    if args.render_pages:
        render_dir = work_dir / "rendered"
        render_dir.mkdir(exist_ok=True)
        render_pages(pdf_path, args.render_pages, render_dir, page_count)

    manifest = {
        "article_id": article_id,
        "paper_title": title,
        "source_pdf": str(pdf_path.relative_to(ROOT)),
        "sha256": sha256(pdf_path),
        "page_count": page_count,
        "text_output": str(text_path.relative_to(ROOT)),
        "rendered_pages": args.render_pages or [],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    with (work_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"已准备 {page_count} 页：{pdf_path.relative_to(ROOT)}")
    print(f"文本：{text_path.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
