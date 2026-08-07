# RAG Application Flow

## What is RAG?

RAG stands for Retrieval-Augmented Generation.

A normal chatbot answers based primarily on information learned during
model training.

A RAG chatbot first searches a collection of external documents and
then gives the retrieved information to the language model.

## Two major stages

### Stage 1: Document ingestion

```text
PDF
 ↓
Extract text
 ↓
Split into chunks
 ↓
Create embeddings
 ↓
Store in ChromaDB
This stage runs through:
python ingest.py
It does not need to run for every question.
Run it again when documents change.
Stage 2: Question answering
User question
 ↓
Create question embedding
 ↓
Search ChromaDB
 ↓
Retrieve matching chunks
 ↓
Build prompt with context
 ↓
Send prompt to Ollama
 ↓
Display answer
This stage runs whenever the user submits a question.
Why chunking is necessary
A PDF may be hundreds of pages long.
Sending the entire PDF to the model would be:
•	slow 
•	expensive for hosted APIs 
•	likely to exceed context limits 
•	less precise 
Chunking divides the document into smaller searchable sections.
Example:
Document:
10,000 characters

Chunk size:
800 characters

Overlap:
150 characters
The overlap helps preserve meaning when a sentence or paragraph crosses
a chunk boundary.
What is an embedding?
An embedding is a numerical representation of text.
Example:
"Employees receive 15 vacation days."
may become a vector conceptually similar to:
[0.15, -0.34, 0.71, ...]
Text with similar meaning generally receives nearby vector
representations.
What is vector search?
The user question is converted into an embedding.
ChromaDB compares that embedding with stored document embeddings and
returns the closest matches.
What is grounding?
Grounding means supplying retrieved document content to the language
model and directing it to answer from that content.
This reduces unsupported answers, but it does not eliminate all model
errors.
Main application files
ingest.py
•	Reads PDF files 
•	Extracts text 
•	Splits pages into chunks 
•	Creates embeddings 
•	Stores chunks in ChromaDB 
rag.py
•	Embeds the user's question 
•	Queries ChromaDB 
•	Formats context 
•	Calls the language model 
•	Returns streamed response chunks 
prompts.py
•	Stores system instructions 
•	Builds the grounded RAG prompt 
app.py
•	Creates the Streamlit interface 
•	Maintains session history 
•	Displays messages 
•	Displays source information 
•	Handles errors 
Important debugging variables
During ingestion, inspect:
pdf_files
pages
documents
embeddings
metadatas
collection.count()
During retrieval, inspect:
question
question_embedding
results
retrieved_chunks
context
messages
During generation, inspect:
model_name
response_stream
chunk
delta
complete_response

---

# 16. VS Code debugger configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug RAG Streamlit App",
      "type": "debugpy",
      "request": "launch",
      "module": "streamlit",
      "args": [
        "run",
        "${workspaceFolder}/app.py"
      ],
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Debug Document Ingestion",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/ingest.py",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
You now have two debugger options:
Debug RAG Streamlit App
Debug Document Ingestion
