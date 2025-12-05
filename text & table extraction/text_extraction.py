import pdfplumber
from pathlib import Path
import pandas as pd
from pdfplumber.utils import intersects_bbox

def extract_text_no_tables(pdf_path: Path):
    """Extract text excluding tables from all pages"""
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # 1. Find tables and get their bounding boxes
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables] if tables else []

            # 2. Filter function - exclude text in table areas
            def outside_tables(obj):
                return not any(intersects_bbox([obj], bbox) for bbox in table_bboxes)

            # 3. Extract CLEAN text (tables excluded)
            clean_page = page.filter(outside_tables)
            full_text = clean_page.extract_text()

            if full_text and full_text.strip():
                all_text.append(full_text.strip())

    return all_text

def consolidate_pdf_texts(input_pdf_folder: str, output_folder: str):
    """Consolidate all pages of each PDF into single text file"""

    input_path = Path(input_pdf_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    pdf_files = list(input_path.glob('*.pdf'))

    print(f"Found {len(pdf_files)} PDFs\n")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        try:
            # Extract text from all pages
            page_texts = extract_text_no_tables(pdf_file)

            if not page_texts:
                print(f"  ⚠ No text extracted from {pdf_file.name}")
                continue

            # Consolidate all pages into one
            consolidated_text = "\n\n".join(page_texts)

            # Save as single file per PDF
            output_file = output_path / f"{pdf_file.stem}-consolidated.txt"
            output_file.write_text(consolidated_text, encoding='utf-8')

            print(f"  ✅ Consolidated {len(page_texts)} pages → {output_file.name}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ All PDFs consolidated to: {output_folder}")

# USAGE
if __name__ == "__main__":
    input_pdf_folder = "/content/KG-RAG"  # Your PDF folder
    output_folder = "./text_extracted"  # Output folder

    consolidate_pdf_texts(input_pdf_folder, output_folder)