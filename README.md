# RAG Document Chatbot

A Streamlit-based Retrieval-Augmented Generation chatbot that answers
questions using locally indexed PDF documents.

## Features

- PDF document ingestion
- Text extraction and chunking
- Sentence Transformer embeddings
- Persistent ChromaDB vector storage
- Semantic document retrieval
- Ollama-based response generation
- Streaming responses
- Source and page references
- Streamlit conversation history

## Architecture

```text
PDF documents
      ↓
Text extraction
      ↓
Text chunking
      ↓
Embedding generation
      ↓
ChromaDB
      ↓
User question
      ↓
Question embedding
      ↓
Similarity search
      ↓
Relevant document chunks
      ↓
Prompt construction
      ↓
Ollama
      ↓
Grounded answer
```

## Requirements

- Python 3.10 or newer
- Ollama
- A downloaded Ollama chat model

## Project setup

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
Python3 -m pip install -r requirements.txt
```

## Ollama setup

Pull the model:

```bash
ollama pull llama3.2:3b
```

Verify that Ollama is available:

```bash
ollama list
```

Test the model:

```bash
ollama run llama3.2:3b
```

## Environment setup

Copy the example configuration:

```bash
cp .env.example .env
```

The default local configuration is:

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2:3b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PATH=chroma_db
CHROMA_COLLECTION=rag_documents
TOP_K_RESULTS=4
```

## Add documents

Copy one or more text-based PDF files into:

```text
data/
```

Example:

```text
data/
├── employee_handbook.pdf
├── benefits_policy.pdf
└── interview_guide.pdf
```

## Ingest the documents

```bash
python ingest.py
```

This creates the local vector database under:

```text
chroma_db/
```

Rerun ingestion whenever documents are added, removed, or changed.

## Run the chatbot

```bash
python3 -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Example questions

- What is the vacation policy?
- How many sick days are available?
- What does the document say about remote work?
- Summarize the employee benefits.
- Which page discusses performance reviews?

## Important limitations

- Scanned PDFs may not contain extractable text.
- The current project does not perform OCR.
- Retrieval quality depends on document quality and chunk settings.
- Language models can still make mistakes.
- Important answers should be verified against the original PDF.

## Rebuild the vector database

Run:

```bash
python3 ingest.py
```

The ingestion program deletes the existing collection and creates a
new one using the current contents of the `data` directory.

## Git workflow

```bash
git status
git add .
git commit -m "Build RAG document chatbot"
git push
```

Do not commit:

- `.env`
- `venv/`
- `chroma_db/`
- secret keys
- confidential documents

## Possible future improvements

- Upload documents through Streamlit
- Support DOCX and TXT files
- Add OCR for scanned PDFs
- Add metadata filters
- Add hybrid keyword and vector search
- Add a reranking model
- Add automated RAG evaluation
- Add user authentication
- Deploy to cloud infrastructure
