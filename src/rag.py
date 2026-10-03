from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"


# ==========================================
# Embedding Model
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# Load PDF Documents
# ==========================================

def load_documents():

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside: {DATA_DIR}"
        )

    print("\nLoading PDF documents...")

    for pdf_path in pdf_files:

        print(f"Loading: {pdf_path.name}")

        loader = PyPDFLoader(
            str(pdf_path)
        )

        pages = loader.load()

        for page in pages:

            # Save file name in document metadata
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    print(
        f"\nLoaded {len(documents)} pages "
        f"from {len(pdf_files)} PDFs."
    )

    return documents


# ==========================================
# Split Documents into Chunks
# ==========================================

def split_documents(documents):

    print(
        "\nSplitting documents into chunks..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# ==========================================
# Local Hugging Face Embeddings
# ==========================================

@lru_cache(maxsize=1)
def get_embeddings():

    print(
        "\nLoading local embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# ==========================================
# Build FAISS Vector Store
# ==========================================

def build_vector_store():

    print(
        "\n================================"
    )

    print(
        "Building MarketMind Vector Store"
    )

    print(
        "================================"
    )

    # Load PDFs
    documents = load_documents()

    # Split PDFs
    chunks = split_documents(
        documents
    )

    # Load embedding model
    embeddings = get_embeddings()

    print(
        "\nCreating FAISS vector store..."
    )

    # Generate embeddings and create FAISS
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create directory
    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save vector database
    vector_store.save_local(
        str(VECTOR_STORE_DIR)
    )

    # Clear cached vector store if rebuilding
    load_vector_store.cache_clear()

    print(
        "\nVector store created successfully."
    )

    return vector_store


# ==========================================
# Load Existing FAISS Store
# ==========================================

@lru_cache(maxsize=1)
def load_vector_store():

    index_file = (
        VECTOR_STORE_DIR / "index.faiss"
    )

    metadata_file = (
        VECTOR_STORE_DIR / "index.pkl"
    )

    if (
        not index_file.exists()
        or not metadata_file.exists()
    ):

        raise FileNotFoundError(
            "Vector store does not exist. "
            "Run: python -m src.rag"
        )

    embeddings = get_embeddings()

    print(
        "\nLoading FAISS vector store..."
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


# ==========================================
# Retrieve Relevant Documents
# ==========================================

def retrieve_documents(
    query: str,
    k: int = 4
):

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


# ==========================================
# Build Retrieved Context
# ==========================================

def retrieve_context(
    query: str,
    k: int = 4
):

    documents = retrieve_documents(
        query=query,
        k=k
    )

    context_parts = []

    sources = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            0
        )

        content = document.page_content

        context_parts.append(
            f"""
DOCUMENT {index}

Source: {source}
Page: {page + 1}

Content:
{content}
"""
        )

        source_label = (
            f"{source} - page {page + 1}"
        )

        if source_label not in sources:
            sources.append(
                source_label
            )

    context = "\n".join(
        context_parts
    )

    return {
        "context": context,
        "sources": sources,
        "documents": documents
    }


# ==========================================
# Retrieval Test
# ==========================================

def test_retrieval():

    query = (
        "What marketing channels work well "
        "for young professionals?"
    )

    print(
        "\n================================"
    )

    print(
        "Testing RAG Retrieval"
    )

    print(
        "================================"
    )

    print(
        f"\nQuery:\n{query}"
    )

    result = retrieve_context(
        query=query
    )

    print(
        "\nRetrieved Context:"
    )

    print(
        result["context"]
    )

    print(
        "\nSources:"
    )

    for source in result["sources"]:

        print(
            f"- {source}"
        )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    build_vector_store()

    test_retrieval()