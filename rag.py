"""
This file handles:
•	embedding the user’s question 
•	searching ChromaDB 
•	formatting retrieved context 
•	calling Ollama 
•	streaming the generated answer 
"""

import os
from collections.abc import Iterator
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from prompts import SYSTEM_PROMPT, build_rag_prompt


load_dotenv()


CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "rag_documents",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))


def create_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client for Ollama or OpenAI."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found in the environment."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_collection():
    """Open the persistent ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    try:
        return client.get_collection(
            name=COLLECTION_NAME
        )
    except ValueError as error:
        raise RuntimeError(
            "The document collection does not exist. "
            "Run 'python ingest.py' before starting the app."
        ) from error


def retrieve_documents(
    question: str,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K_RESULTS,
) -> list[dict[str, Any]]:
    """Retrieve document chunks semantically related to a question."""

    if not question.strip():
        return []

    collection = get_collection()

    available_documents = collection.count()

    if available_documents == 0:
        return []

    result_count = min(top_k, available_documents)

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    retrieved_chunks: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown source",
                ),
                "page": metadata.get("page"),
                "chunk": metadata.get("chunk"),
                "distance": distance,
            }
        )

    return retrieved_chunks


def format_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Convert retrieved chunks into context for the LLM."""

    if not retrieved_chunks:
        return "No relevant document context was retrieved."

    context_sections: list[str] = []

    for index, item in enumerate(
        retrieved_chunks,
        start=1,
    ):
        source = item["source"]
        page = item["page"]
        text = item["text"]

        context_sections.append(
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text: {text}"
        )

    return "\n\n".join(context_sections)


def build_chat_messages(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Create the messages sent to the language model."""

    context = format_context(retrieved_chunks)
    rag_question = build_rag_prompt(
        question=question,
        context=context,
    )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if conversation_history:
        # Limit the amount of prior chat sent to the model.
        recent_messages = conversation_history[-6:]

        for message in recent_messages:
            if message["role"] not in {
                "user",
                "assistant",
            }:
                continue

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

    messages.append(
        {
            "role": "user",
            "content": rag_question,
        }
    )

    return messages


def create_response_stream(
    llm_client: OpenAI,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
) -> Iterator[Any]:
    """Create a streamed response from the language model."""

    model_name = os.getenv(
        "OPENAI_MODEL",
        "llama3.2:3b",
    )

    messages = build_chat_messages(
        question=question,
        retrieved_chunks=retrieved_chunks,
        conversation_history=conversation_history,
    )

    return llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        stream=True,
    )


def get_source_labels(
    retrieved_chunks: list[dict[str, Any]],
) -> list[str]:
    """Return unique, readable source labels."""

    source_labels: list[str] = []

    for item in retrieved_chunks:
        source = item["source"]
        page = item["page"]

        if page is not None:
            label = f"{source}, page {page}"
        else:
            label = source

        if label not in source_labels:
            source_labels.append(label)

    return source_labels
