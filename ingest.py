"""
Document-ingestion pipeline for the RAG chatbot.

This program:

1. Finds PDF files in the data directory.
2. Extracts readable text from each PDF page.
3. Splits the text into overlapping chunks.
4. Creates embeddings for the chunks.
5. Stores the chunks, embeddings, and metadata in ChromaDB.
"""

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()


DATA_DIRECTORY = Path("data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "rag_documents",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """Extract text from every readable page of a PDF."""

    reader = PdfReader(pdf_path)
    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = page.extract_text()

        if not page_text:
            print(
                f"Skipping page {page_number} in "
                f"{pdf_path.name}: no readable text."
            )
            continue

        cleaned_text = " ".join(page_text.split())

        if not cleaned_text:
            print(
                f"Skipping page {page_number} in "
                f"{pdf_path.name}: extracted text was empty."
            )
            continue

        pages.append(
            {
                "source": pdf_path.name,
                "page": page_number,
                "text": cleaned_text,
            }
        )

    print(
        f"Extracted {len(pages)} readable page(s) "
        f"from {pdf_path.name}."
    )

    return pages


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_chunk_id(
    source: str,
    page: int,
    chunk_number: int,
    text: str,
) -> str:
    """Create a stable unique ID for a document chunk."""

    raw_value = (
        f"{source}:{page}:{chunk_number}:{text}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


def prepare_documents() -> tuple[
    list[str],
    list[str],
    list[dict[str, Any]],
]:
    """Read PDFs and prepare IDs, chunks, and metadata."""

    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"The data directory does not exist: "
            f"{DATA_DIRECTORY.resolve()}"
        )

    pdf_files = sorted(
        DATA_DIRECTORY.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found in the data directory."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        print()
        print(f"Reading: {pdf_path.name}")

        pages = extract_pdf_pages(pdf_path)

        if not pages:
            print(
                f"No readable pages were found in "
                f"{pdf_path.name}."
            )
            continue

        for page_data in pages:
            page_chunks = split_text(
                page_data["text"]
            )

            print(
                f"Page {page_data['page']}: "
                f"created {len(page_chunks)} chunk(s)."
            )

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1,
            ):
                chunk_id = create_chunk_id(
                    source=page_data["source"],
                    page=page_data["page"],
                    chunk_number=chunk_number,
                    text=chunk,
                )

                ids.append(chunk_id)
                documents.append(chunk)

                metadatas.append(
                    {
                        "source": page_data["source"],
                        "page": page_data["page"],
                        "chunk": chunk_number,
                    }
                )

    return ids, documents, metadatas


def recreate_collection(
    client: chromadb.PersistentClient,
):
    """
    Delete the existing collection when present and create
    a clean collection.

    On the first run, the collection does not exist. ChromaDB
    raises NotFoundError in that situation, which is expected.
    """

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Deleted existing collection: "
            f"{COLLECTION_NAME}"
        )

    except (NotFoundError, ValueError):
        print(
            f"Collection '{COLLECTION_NAME}' "
            "does not exist yet. Creating it now."
        )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "PDF chunks for the RAG chatbot"
            ),
        },
    )


def ingest_documents() -> None:
    """Run the complete document-ingestion process."""

    print("Preparing documents...")

    ids, documents, metadatas = (
        prepare_documents()
    )

    if not documents:
        raise ValueError(
            "The PDF files did not contain readable text. "
            "The files may be scanned images or malformed."
        )

    print()
    print(
        f"Prepared {len(documents)} chunk(s)."
    )

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Opening ChromaDB at: {CHROMA_PATH}"
    )

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = recreate_collection(
        chroma_client
    )

    print("Saving chunks and embeddings...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    stored_count = collection.count()

    print()
    print(
        "Document ingestion completed successfully."
    )
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {stored_count}")

    if stored_count != len(documents):
        print(
            "Warning: the number of stored chunks "
            "does not match the number prepared."
        )


if __name__ == "__main__":
    try:
        ingest_documents()

    except FileNotFoundError as error:
        print()
        print("Document ingestion failed.")
        print("Reason: file or directory not found.")
        print(f"Error detail: {error}")

    except ValueError as error:
        print()
        print("Document ingestion failed.")
        print("Reason: invalid or unreadable document data.")
        print(f"Error detail: {error}")

    except Exception as error:
        print()
        print("Document ingestion failed.")
        print(
            f"Error type: "
            f"{type(error).__name__}"
        )
        print(f"Error detail: {error}")
        raise