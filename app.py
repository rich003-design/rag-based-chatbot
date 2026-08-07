"""Streamlit interface for the RAG document chatbot."""

from typing import Any

import streamlit as st

from rag import (
    create_llm_client,
    create_response_stream,
    get_source_labels,
    load_embedding_model,
    retrieve_documents,
)


WELCOME_MESSAGE = (
    "Hello! I am your document assistant. "
    "Ask me a question about the PDF documents that have "
    "been added to this project."
)


def initialize_session() -> None:
    """Initialize the Streamlit conversation history."""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "sources": [],
            }
        ]


def reset_conversation() -> None:
    """Reset the conversation to the welcome message."""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "sources": [],
        }
    ]


@st.cache_resource
def get_embedding_model():
    """Load and cache the embedding model."""

    return load_embedding_model()


@st.cache_resource
def get_llm_client():
    """Create and cache the LLM client."""

    return create_llm_client()


def display_sources(sources: list[str]) -> None:
    """Display source labels below an assistant response."""

    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            st.write(f"- {source}")


def display_streamed_response(
    response_stream: Any,
) -> str:
    """Display streamed text and return the complete response."""

    placeholder = st.empty()
    complete_response = ""

    for chunk in response_stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content

        if delta:
            complete_response += delta
            placeholder.markdown(
                complete_response + "▌"
            )

    if not complete_response:
        complete_response = (
            "I could not generate a text response. "
            "Please try again."
        )

    placeholder.markdown(complete_response)

    return complete_response


st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="centered",
)

initialize_session()

st.title("📚 RAG Document Chatbot")

st.caption(
    "Ask questions about your indexed PDF documents."
)

with st.sidebar:
    st.header("About")

    st.write(
        """
        This chatbot uses Retrieval-Augmented Generation:

        1. Your question is converted into an embedding.
        2. ChromaDB finds relevant PDF sections.
        3. The sections are sent to the language model.
        4. The model creates a grounded answer.
        """
    )

    st.divider()

    st.subheader("Setup")

    st.code(
        "python ingest.py\n"
        "python -m streamlit run app.py",
        language="bash",
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(
                message.get("sources", [])
            )


user_prompt = st.chat_input(
    "Ask a question about your documents..."
)


if user_prompt:
    user_message = {
        "role": "user",
        "content": user_prompt,
        "sources": [],
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.status(
                "Searching the documents...",
                expanded=False,
            ) as status:
                embedding_model = get_embedding_model()

                retrieved_chunks = retrieve_documents(
                    question=user_prompt,
                    embedding_model=embedding_model,
                )

                status.update(
                    label=(
                        f"Retrieved "
                        f"{len(retrieved_chunks)} "
                        f"document sections."
                    ),
                    state="complete",
                )

            llm_client = get_llm_client()

            response_stream = create_response_stream(
                llm_client=llm_client,
                question=user_prompt,
                retrieved_chunks=retrieved_chunks,
                conversation_history=(
                    st.session_state.messages[:-1]
                ),
            )

            assistant_response = (
                display_streamed_response(
                    response_stream
                )
            )

            source_labels = get_source_labels(
                retrieved_chunks
            )

            display_sources(source_labels)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                    "sources": source_labels,
                }
            )

        except Exception as error:
            st.error(
                "I could not generate a response. "
                "Check that Ollama is running and that "
                "the documents were ingested."
            )

            st.caption(
                f"Technical detail: "
                f"{type(error).__name__}"
            )

            # Use this temporarily while debugging:
            # st.exception(error)
