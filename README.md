**Financial RAG Pipeline for Multi-PDF Analysis (Text + Tables)**

<img width="1405" height="611" alt="image" src="https://github.com/user-attachments/assets/aad7a5df-58a0-4142-88f3-f487a6d59db4" />


---

# **📘 Project Overview**

This project is a **Financial RAG (Retrieval-Augmented Generation) System** built for analyzing quarterly financial reports of major IT companies across **2022–2023**.
It processes raw PDFs, extracts tables and text, stores structured chunks in ChromaDB, retrieves relevant segments using embeddings, and generates LLM-powered financial summaries through **Llama 3.2 (3B)** via Ollama.

---

# **📂 Dataset Summary**

* **20 financial-report PDFs**
* Each report contains:

  * Financial information for **Q1–Q3** across **2022 & 2023**
  * Data for **5 IT companies**
* Each PDF contains:

  * Multiple complex financial tables
  * Narrative business performance text

To preserve both structured and unstructured information, the project extracts:

* **Pure text content**
* **Individual tables** as CSV → JSON

---

# **🧠 System Architecture**

The workflow is divided into clean stages:

---

## **1️⃣ PDF Table Extraction**

* Used **Docling** to extract **all tables** from each PDF.
* Each table is exported as an individual **CSV file**.
* Folder structure:

  ```
  pdf_01/
      table_01.csv
      table_02.csv
      ...
  pdf_20/
      table_01.csv
      ...
  ```
* Total tables extracted: **827 CSVs**

Tables are then converted into **JSON** for:

* cleaner structure
* better parsing
* preventing loss of numeric context

---

## **2️⃣ PDF Text Extraction**

* Used **pdfplumber** to extract pure narrative text.
* Each PDF receives one text file containing:

  * business commentary
  * operational updates
  * financial explanations

The extracted text becomes the basis for embedding + chunking.

---

## **3️⃣ Chunking Pipeline**

Used **LangChain RecursiveCharacterTextSplitter**:

* `chunk_size = 800`
* `chunk_overlap = 100`

This creates meaningful segments without breaking semantics.

Total chunks created: **5291**

---

## **4️⃣ Metadata Injection**

Each chunk receives detailed metadata for traceability:

```json
{
    "pdf_id": "company_2022_Q1",
    "chunk_id": "company_2022_Q1_chunk_17",
    "num_tables": <tables_in_pdf>,
    "company_name": "<extracted name>",
    "year": "2022",
    "quarter": "Q1"
}
```

Metadata enables:

* filtering by company
* filtering by year/quarter
* retrieving exact source chunks
* showing attribution in UI

---

## **5️⃣ Embedding & Vector Indexing**

Embedding model:

* **all-MiniLM-L6-v2** (Sentence Transformers)
  → chosen for **speed** & **high recall**

Vector store:

* **ChromaDB (PersistentClient)**
* Stores both:

  * text chunks
  * JSON tables

Two types of collections are indexed:

1. **Text collection**
2. **Table metadata collection**

Each record includes:

* vector embedding
* content
* metadata

---

## **6️⃣ Retrieval (ChromaDB Query)**

For every user query, the system:

* Runs **metadata extraction** using Llama
  (extracts company name)
* Builds dynamic `where` filters
* Runs:

  ```python
  collection.query(
      query_texts=[query],
      n_results=5,
      include=["documents", "metadatas", "distances"],
      where=<built_filter>
  )
  ```

Retrieves top chunks with:

* documents
* metadata
* distance scores (relevance)

---

## **7️⃣ LLM Summarization**

Model:

* **Llama 3.2 (3B) via Ollama**

Prompt:

* Provide a **factual summary**
* Use **markdown** and **tables**
* Answer only based on retrieved chunks

The model receives:

* query
* 5 most relevant chunks
* produces concise financial insights

---

## **8️⃣ Flask Web Application**

A lightweight Flask app wraps the entire pipeline:

* Accepts user queries
* Runs retrieval + summary generation
* Returns:

  * HTML-rendered summary
  * Source snippets
  * Relevance score
  * Metadata (PDF ID, chunk ID)

Designed for interactive, user-friendly exploration.

---

# **🚀 Features**

* Full financial RAG pipeline
* Text + table extraction preserved
* 5 company metadata filtering
* 800-character chunking for balanced retrieval
* SentenceTransformer embeddings
* ChromaDB persistent indexing
* Llama 3.2 summarization
* Flask UI with:

  * Summary
  * Source highlight
  * Snippets
  * Distance-based relevance tags

---

# **🛠 Tech Stack**

| Component            | Tool                  |
| -------------------- | --------------------- |
| LLM                  | Llama 3.2:3B (Ollama) |
| Embeddings           | all-MiniLM-L6-v2      |
| Vector DB            | ChromaDB              |
| Table Extraction     | Docling               |
| Text Extraction      | pdfplumber            |
| Backend              | Flask                 |
| Chunking & Pipelines | LangChain             |
| Storage              | JSON, CSV, SQLite     |

---

# **📈 Summary**

This project delivers a complete, production-ready **Financial RAG Engine** for analyzing multi-company quarterly financials with:

* accurate retrieval
* clean metadata usage
* structured table preservation
* clear UI presentation

Perfect for:

* financial research
* business analysis
* RAG experimentation
* LLM-powered report automation


