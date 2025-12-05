
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

CHROMA_PATH = "enhanced_rag_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

# Load existing ChromaDB index
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = chroma_client.get_collection(
    name="financial_reports_langchain",
    embedding_function=ef
)

print(f"✅ Loaded index with {collection.count()} chunks\n")

# Initialize Ollama LLM client
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

def extract_metadata_llm(query: str):
    """Use LLM to extract company_name as JSON from query"""
    prompt = PromptTemplate.from_template("""
Analyze this financial query and extract metadata for filtering documents.

Query: {query}

Extract ONLY these fields as JSON (use null if not mentioned). Return single values, not lists:

{{
  "company_name": "Amazon" or "Microsoft" or "Apple" or "Intel" or "NVIDIA" or null
}}

Respond with valid JSON only, no extra text:
""")

    parser = JsonOutputParser()
    chain = prompt | llm | parser
    metadata = chain.invoke({"query": query})
    return metadata

def build_where_filter(metadata_filters: dict):
    """Build ChromaDB 'where' filter using $and operator for multiple conditions"""
    conditions = []

    for key in ["company_name"]:
        value = metadata_filters.get(key)
        if value and value != "null" and value is not None:
            if isinstance(value, list):
                # Use $in if multiple values, $eq if single
                if len(value) == 1:
                    # Convert company_name to title case to match index format
                    val = value[0].title() if key == "company_name" else value[0]
                    conditions.append({key: {"$eq": val}})
                else:
                    # Convert company names to title case
                    vals = [v.title() if key == "company_name" else v for v in value if v != "null"]
                    conditions.append({key: {"$in": vals}})
            else:
                # Convert company_name to title case to match index format
                val = value.title() if key == "company_name" else value
                conditions.append({key: {"$eq": val}})

    # Return None if no conditions, otherwise wrap with $and
    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

def pretty_print_summary(summary: str):
    summary = summary.replace("\\n", "\n").replace("* ", "  * ").replace("- ", "  - ")
    print("\n### Summary:\n")
    print(summary)
    print("\n---\n")



def query_with_summarization(query, n_results=5, return_html=True):

    # Extract metadata
    metadata_filters = extract_metadata_llm(query)
    where_filter = build_where_filter(metadata_filters)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=where_filter if where_filter else None
    )

    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    # LLM Summary
    context = "\n\n".join(docs)

    prompt_template = PromptTemplate.from_template("""
You are a financial analyst. Given the following documents, provide a concise, factually correct summary answering the question: {query}

Documents:
{context}

Answer in markdown with tables.

Summary:
""")

    prompt = prompt_template.format(query=query, context=context)
    summary_md = llm.invoke(prompt).content

    if not return_html:
        pretty_print_summary(summary_md)
        return summary_md


    # ---- HTML BUILDING ----
    import markdown
    summary_html = markdown.markdown(summary_md, extensions=['tables', 'fenced_code'])

    sources_html = ""
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
        snippet = doc[:300].replace("\n", " ")
        sources_html += f"""
            <div class="source-card">
                <h4>Source {i}</h4>
                <p><b>PDF ID:</b> {meta.get("pdf_id")}</p>
                <p><b>Chunk ID:</b> {meta.get("chunk_id")}</p>
                <p><b>Distance:</b> {dist:.3f}</p>
                <p><b>Snippet:</b> {snippet}...</p>
            </div>
        """

    final_html = f"""
        <div class='answer-section'>
            <h2>Summary</h2>
            {summary_html}
            <hr>
            <h2>Sources</h2>
            {sources_html}
        </div>
    """

    return final_html
