from pathlib import Path
import pandas as pd

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# 1) Configure PDF pipeline (enable table structure)
pipeline_options = PdfPipelineOptions(
    do_table_structure=True  # turn on table detection/structure [web:28]
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

input_doc_path = Path("/content/2022 Q3 AMZN.pdf")

# 2) Convert the PDF to a Docling document
conv_res = doc_converter.convert(input_doc_path)  # returns ConversionResult[web:21][web:28]
doc = conv_res.document                          # DoclingDocument with tables/text/pictures[web:29]

# 3) Iterate detected tables and export to DataFrame / CSV / HTML
out_dir = Path("tables_out_2")
out_dir.mkdir(exist_ok=True)

for table_ix, table in enumerate(doc.tables):  # all tables in the document[web:29]
    # to pandas DataFrame
    table_df: pd.DataFrame = table.export_to_dataframe(doc=doc)  # structured table[web:20][web:27]

    print(f"\n## Table {table_ix}")
    print(table_df.head())

    stem = input_doc_path.stem

    # Save as CSV
    csv_path = out_dir / f"{stem}-table-{table_ix}.csv"
    table_df.to_csv(csv_path, index=False)

    # Save as HTML
    html_path = out_dir / f"{stem}-table-{table_ix}.html"
    html_html = table.export_to_html(doc=doc)  # HTML representation of the table[web:20][web:27]
    html_path.write_text(html_html, encoding="utf-8")
