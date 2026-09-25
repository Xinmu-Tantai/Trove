from pathlib import Path
from pypdf import PdfReader

source = Path("sources/2406.09246.pdf")
output = Path("work/2406.09246/paper.txt")

reader = PdfReader(source)
with output.open("w", encoding="utf-8") as file:
    for number, page in enumerate(reader.pages, start=1):
        file.write(f"\n\n===== PAGE {number} =====\n\n")
        file.write(page.extract_text(extraction_mode="layout") or "")

print(f"pages={len(reader.pages)}")
