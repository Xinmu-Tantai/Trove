import sys
from pathlib import Path

sys.path.insert(0, "work/2406.09246/vendor")
import fitz

source = Path("sources/2406.09246.pdf")
output_dir = Path("work/2406.09246/rendered")
output_dir.mkdir(parents=True, exist_ok=True)

document = fitz.open(source)
for page_number in (1, 4, 7, 9, 10, 37):
    page = document[page_number - 1]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pixmap.save(output_dir / f"page-{page_number}.png")
