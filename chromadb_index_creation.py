import json
import re
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

JSON_FOLDER = "/content/rag_json_output"
TXT_FOLDER = "/content/KG_text_ext"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "/content/rag_index"  

def extract_pdf_id(filename: str) -> str:
    """Extract PDF ID: tables_2022 Q3 AAPL-table-0.json -> AAPL_2022_Q3"""
    # Handle both "tables_2022 Q3 AAPL" and "tables10_2023 Q1 NVDA" formats
    match = re.search(r'(\d{4})\s(Q\d)\s([A-Z]+)', filename)
    if match:
        return f"{match.group(3)}_{match.group(1)}_{match.group(2)}"
    print(f"⚠️  Could not extract ID from: {filename}")
    return None

def load_json_table(json_path: Path):
    """Load JSON table to text"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Data is nested under 'text' key
        table_data = data.get('text', {})
        rows = table_data.get('rows', [])
        columns = table_data.get('columns', [])

        if not rows or not columns:
            return None

        # Convert to DataFrame and stringify
        df = pd.DataFrame(rows, columns=columns)
        text_rows = [' | '.join([str(x) for x in row if pd.notna(x)]) for _, row in df.iterrows()]
        text_rows = [r.strip() for r in text_rows if r.strip()]

        if not text_rows:
            return None

        pdf_id = extract_pdf_id(json_path.name)
        return {
            'pdf_id': pdf_id,
            'table_idx': json_path.stem.split('-')[-1],
            'text': '\n'.join(text_rows),
            'row_count': len(text_rows)
        }
    except Exception as e:
        print(f"❌ {json_path.name}: {e}")
        return None

def load_txt_content(txt_path: Path):
    """Load TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return None

        # Extract PDF ID from filename like "2023 Q2 AAPL-consolidated.txt"
        match = re.search(r'(\d{4})\s(Q\d)\s(\w+)', txt_path.name)
        pdf_id = f"{match.group(3)}_{match.group(1)}_{match.group(2)}" if match else txt_path.stem

        return {'pdf_id': pdf_id, 'text': content}
    except Exception as e:
        print(f"❌ {txt_path.name}: {e}")
        return None

# Mapping ticker to full company name
COMPANY_MAP = {
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'MSFT': 'Microsoft',
    'NVDA': 'Nvidia',
    'INTC': 'Intel'
}

def get_company_name(pdf_id: str) -> str:
    ticker = pdf_id.split('_')[0]
    return COMPANY_MAP.get(ticker, ticker)  # fallback to ticker if unknown

# Inside your for-loop in process_all_documents() after splitting chunks

def process_all_documents():
    """Process JSONs + TXTs with LangChain chunking"""
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " | ", " ", ""],
        length_function=len,
    )

    # Load all JSON tables grouped by PDF ID
    json_files = list(Path(JSON_FOLDER).rglob("*.json"))
    pdf_groups = {}

    print(f"📂 Loading {len(json_files)} JSON tables...")
    for json_path in tqdm(json_files, desc="JSON"):
        table_data = load_json_table(json_path)
        if table_data:
            pdf_id = table_data['pdf_id']
            if pdf_id not in pdf_groups:
                pdf_groups[pdf_id] = []
            pdf_groups[pdf_id].append(table_data)

    # Process each TXT + its tables
    txt_files = list(Path(TXT_FOLDER).glob("*.txt"))
    print(f"\n📂 Processing {len(txt_files)} TXT files...")

    for txt_path in tqdm(txt_files, desc="TXT"):
        txt_data = load_txt_content(txt_path)
        if not txt_data:
            continue

        pdf_id = txt_data['pdf_id']

        # Combine TXT + tables
        full_doc = f"PDF: {pdf_id}\n\nTEXT:\n{txt_data['text']}\n\nTABLES:\n"

        if pdf_id in pdf_groups:
            for table in pdf_groups[pdf_id]:
                full_doc += f"Table {table['table_idx']} ({table['row_count']} rows):\n{table['text']}\n\n"

        # Chunk the document
        doc_chunks = splitter.split_text(full_doc)

        for i, chunk in enumerate(doc_chunks):
            if chunk.strip():
                chunks.append({
            'content': chunk,
            'metadata': {
                'pdf_id': pdf_id,
                'chunk_id': f"{pdf_id}_chunk_{i}",
                'num_tables': len(pdf_groups.get(pdf_id, [])),
                'company_name': get_company_name(pdf_id),
                'year': pdf_id.split('_')[1],
                'quarter': pdf_id.split('_')[2]
            }
        })

    print(f"\n✅ Created {len(chunks)} chunks from {len(pdf_groups)} PDFs")
    return chunks

def create_vector_index(chunks):
    """Embed and index with ChromaDB"""
    if not chunks:
        print("❌ No chunks to index!")
        return

    print("\n🔄 Creating embeddings...")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = chroma_client.get_or_create_collection(
        name="financial_reports_langchain",
        embedding_function=ef
    )

    # Batch add
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i:i+batch_size]
        collection.add(
            documents=[c['content'] for c in batch],
            metadatas=[c['metadata'] for c in batch],
            ids=[c['metadata']['chunk_id'] for c in batch]
        )

    print(f"✅ Index ready: {CHROMA_PATH}")
    print(f"📊 Total chunks indexed: {collection.count()}")

if __name__ == "__main__":
    chunks = process_all_documents()
    if chunks:
        create_vector_index(chunks)
        print("\n🎉 Complete!")
    else:
        print("\n❌ Failed: No chunks created")