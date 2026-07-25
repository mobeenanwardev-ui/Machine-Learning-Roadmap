"""
Simple RAG pipeline based on the course exercise.

Workflow:
1. Load TXT, PDF, and DOCX documents.
2. Split them into chunks.
3. Convert chunks into embeddings.
4. Store embeddings + original text in Pinecone.
5. Embed a user question.
6. Retrieve semantically similar chunks.
7. Build an augmented prompt.
8. Generate an answer with a local Llama model through Ollama.

Important:
RAG does not retrain the LLM. It retrieves external information and supplies
that information as context at query time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import docx
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Newer LangChain versions expose text splitters from a separate package.
# The fallback keeps the script close to the import used in the course notebook.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter


DOCS_DIR = Path(__file__).parent / "docs"
INDEX_NAME = "rag-tutorial"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
LLM_MODEL = "llama3.2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10
BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# 1. Load documents
# ---------------------------------------------------------------------------


def extract_text_from_file(file_path: Path) -> Optional[str]:
    """Extract text from a TXT, PDF, or DOCX file."""

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        text_parts: list[str] = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    if suffix == ".docx":
        document = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return None


def load_documents(folder_path: Path = DOCS_DIR) -> list[str]:
    """Load all supported documents from the docs directory."""

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Document folder not found: {folder_path}\n"
            "Create a 'docs' folder beside this script and add TXT, PDF, or DOCX files."
        )

    documents: list[str] = []

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue

        text = extract_text_from_file(file_path)
        if text and text.strip():
            documents.append(text)

    if not documents:
        raise ValueError(
            f"No supported documents were found in {folder_path}. "
            "Add at least one TXT, PDF, or DOCX file."
        )

    return documents


# ---------------------------------------------------------------------------
# 2. Chunk documents
# ---------------------------------------------------------------------------


def chunk_documents(documents: list[str]) -> list[str]:
    """Split large document text into smaller overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_text(" ".join(documents))
    print(f"Loaded {len(chunks)} text chunks.")
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding model + Pinecone
# ---------------------------------------------------------------------------


def create_pinecone_index(api_key: str):
    """Create or connect to the Pinecone index used by the exercise."""

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(INDEX_NAME)


def embed_and_store_chunks(
    chunks: list[str],
    embedder: SentenceTransformer,
    index,
) -> None:
    """Embed chunks and store vector + original text metadata in Pinecone."""

    # The vector is used for semantic search.
    # The metadata stores the human-readable chunk that we give back to the LLM.
    vectors = [
        {
            "id": str(i),
            "values": embedder.encode(chunk).tolist(),
            "metadata": {"text": chunk},
        }
        for i, chunk in enumerate(chunks)
    ]

    # Upsert = insert a new ID or update the existing ID.
    for start in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[start : start + BATCH_SIZE]
        index.upsert(batch)

    print(f"Inserted {len(vectors)} vectors into '{INDEX_NAME}'.")


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------


def retrieve_context(
    question: str,
    embedder: SentenceTransformer,
    index,
    top_k: int = TOP_K,
) -> str:
    """Embed a question, retrieve similar chunks, and return their original text."""

    question_vector = embedder.encode(question).tolist()

    results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
    )

    contexts = [
        match["metadata"]["text"]
        for match in results["matches"]
        if match.get("metadata") and match["metadata"].get("text")
    ]

    return "\n\n".join(contexts)


# ---------------------------------------------------------------------------
# 5. Augment prompt + generate answer
# ---------------------------------------------------------------------------


def build_prompt(question: str, context_text: str) -> str:
    """Combine retrieved context, the user question, and an instruction."""

    return f"""Answer the question based only on the context below.

Context:
{context_text}

Question: {question}
Answer:"""


def answer_question(question: str, context_text: str, model: OllamaLLM) -> str:
    """Generate the final answer using the augmented prompt."""

    prompt = build_prompt(question, context_text)
    return model.invoke(prompt)


# ---------------------------------------------------------------------------
# 6. Complete RAG workflow
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set.\n"
            "Example: export PINECONE_API_KEY='your-key'"
        )

    # Indexing phase
    documents = load_documents()
    chunks = chunk_documents(documents)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    index = create_pinecone_index(api_key)
    embed_and_store_chunks(chunks, embedder, index)

    # Generation phase
    model = OllamaLLM(model=LLM_MODEL)

    question = input("\nAsk a question about your documents: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    context_text = retrieve_context(question, embedder, index)

    print("\n=== Retrieved context ===")
    print(context_text)

    response = answer_question(question, context_text, model)

    print("\n=== RAG answer ===")
    print(response)

    print(
        "\nMemory rule: "
        "CHUNK -> EMBED -> STORE -> QUESTION -> EMBED -> "
        "RETRIEVE -> CONTEXT -> PROMPT -> LLM"
    )


if __name__ == "__main__":
    main()
