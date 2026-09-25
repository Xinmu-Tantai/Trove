#!/usr/bin/env python3
"""Crop a rectangular region from one PDF page into a PNG image."""

from __future__ import annotations

import argparse
from pathlib import Path


def rectangle(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(number.strip()) for number in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("区域应为 x0,y0,x1,y1") from error
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        raise argparse.ArgumentTypeError("区域应为 x0,y0,x1,y1，且右下角在左上角之后")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PDF 单页的矩形区域裁剪为 PNG。")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--page", required=True, type=int, help="从 1 开始的页码")
    parser.add_argument("--rect", required=True, type=rectangle, help="x0,y0,x1,y1，单位为 PDF point")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    if args.page < 1 or args.scale <= 0:
        parser.error("page 和 scale 必须为正数")

    try:
        import fitz
    except ImportError as error:
        raise SystemExit("需要 PyMuPDF：python3 -m pip install PyMuPDF") from error

    document = fitz.open(args.pdf)
    if args.page > len(document):
        parser.error(f"页码超出范围：该 PDF 共 {len(document)} 页")
    pixmap = document[args.page - 1].get_pixmap(
        matrix=fitz.Matrix(args.scale, args.scale), clip=fitz.Rect(args.rect), alpha=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(args.output)


if __name__ == "__main__":
    main()
