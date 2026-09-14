# Implementation Guide — PDF AI Assistant

This document explains how the current PDF RAG chatbot is implemented, how the major components interact, and why the application behaves differently for PDF questions, general chat, and remembered user information.

---

## 1. Technology Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | HTML/CSS/JavaScript | UI, upload, chat, stats, memory controls |
| API | FastAPI | HTTP endpoints and validation |
| PDF parser | pypdf | Extract page text |
| Embeddings | Google Gemini `gemini-embedding-001` | Convert text/questions to vectors |
| Chat model | Google Gemini `gemini-3.1-flash-lite` | Generate grounded or general responses |
| Vector math | NumPy | Cosine similarity |
| Persistence | JSON files | PDF index + lightweight memory |
| Configuration | python-dotenv | Load `backend/.env` |
| Server | Uvicorn | Run FastAPI |

No separate frontend build system is required.

---

## 2. Runtime Components

The backend is divided into three Python modules:

```text
backend/app/
├── config.py
├── main.py
└── rag.py
```

### `config.py`

Centralizes configuration and paths.

### `main.py`

Defines FastAPI routes, request models, file validation, and application startup behavior.

### `rag.py`

Contains the core RAG implementation, vector store, memory store, PDF extraction, chunking, retrieval, validation, and Gemini calls.

---

## 3. Configuration Implementation

`config.py` determines the backend directory with:

```python
BACKEND_DIR = Path(__file__).resolve().parents[1]
```

The `.env` file is explicitly loaded from:

```text
backend/.env
```

The application supports both:

```text
GOOGLE_API_KEY
```

and:

```text
GEMINI_API_KEY
```

with `GOOGLE_API_KEY` taking priority.

Important configurable values are:

```text
CHAT_MODEL
EMBEDDING_MODEL
TOP_K
CHUNK_SIZE
CHUNK_OVERLAP
MAX_FILE_SIZE_MB
RELEVANCE_THRESHOLD
```

Runtime data is stored under:

```text
backend/data/
```

with:

```text
index.json
memory.json
```

---

## 4. FastAPI Application

`main.py` creates the application:

```python
app = FastAPI(title="PDF AI Assistant", version="1.0.0")
```

A global `rag` variable is lazily initialized.

### Why lazy initialization?

The Gemini client and local stores do not need to be constructed until an API operation actually requires them. It also allows `/api/health` to report API-key status without immediately making a Gemini request.

The helper:

```python
def get_rag():
```

checks for the API key and constructs `RAG()` on first use.

---

## 5. Frontend Serving and Cache Handling

The root route returns:

```python
return FileResponse(
    config.FRONTEND_FILE,
    headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
)
```

This is useful during local UI development because browsers can otherwise continue showing an older `index.html`.

The frontend path is resolved to:

```text
project/frontend/index.html
```

through `config.FRONTEND_FILE`.

---

## 6. PDF Upload Implementation

Endpoint:

```text
POST /api/upload
```

The endpoint accepts:

```python
files: list[UploadFile] = File(...)
```

For every uploaded file it performs:

1. Filename validation.
2. `.pdf` extension validation.
3. Byte-size validation.
4. PDF indexing.
5. Chunk counting.
6. Response generation.

The maximum file size is controlled by:

```text
MAX_FILE_SIZE_MB
```

The actual byte limit is calculated as:

```python
max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
```

---

## 7. PDF Text Extraction

`RAG.index_pdf()` receives the filename and raw PDF bytes.

```python
reader = PdfReader(BytesIO(data))
```

Pages are processed one at a time:

```python
for page_no, page in enumerate(reader.pages, start=1):
```

The page text is extracted with:

```python
page.extract_text()
```

This preserves page metadata so the final answer can identify the page where a retrieved chunk came from.

Metadata is stored as:

```json
{
  "file_name": "example.pdf",
  "page": 2
}
```

---

## 8. Text Normalization

The `normalize()` function:

- Removes null characters.
- Collapses repeated whitespace within lines.
- Removes empty lines.
- Produces cleaner text before chunking.

This improves consistency for embeddings and retrieval.

---

## 9. Chunking Strategy

The project uses configurable character-based chunks.

Defaults:

```text
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
```

The overlap ensures that information near a chunk boundary can still appear in neighboring chunks.

The implementation attempts to end chunks at a useful boundary when possible:

1. Newline.
2. Sentence boundary.
3. Space.

If a natural boundary is far too early, the configured chunk size is retained.

A safety check prevents invalid configuration:

```python
if overlap >= size:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
```

---

## 10. Embedding Implementation

The application uses Gemini's embedding API:

```python
self.client.models.embed_content(
    model=config.EMBEDDING_MODEL,
    contents=texts,
)
```

Each chunk becomes a numeric vector.

During a question, the question itself is embedded using the same embedding model. This makes vector comparison meaningful because document chunks and the query exist in the same embedding space.

---

## 11. Local Vector Store

The `Store` class provides a minimal persistent vector database.

Each indexed item looks conceptually like:

```json
{
  "text": "retrieved PDF text...",
  "vector": [0.01, -0.04, 0.12],
  "metadata": {
    "file_name": "example.pdf",
    "page": 1
  }
}
```

The complete list is saved to:

```text
backend/data/index.json
```

This is intentionally simple for an assignment. It avoids requiring a separate vector database.

---

## 12. Cosine Similarity

When a question is received, the question vector is compared with every stored vector.

The implementation calculates:

```text
cosine_similarity(q, v) = (q · v) / (||q|| ||v||)
```

Higher values indicate that the vectors are more similar.

The store sorts results by descending score and returns the first `TOP_K` results.

Default:

```text
TOP_K=5
```

---

## 13. Why a Relevance Threshold Exists

Pure vector similarity can always return something, even when the user's question has nothing to do with the documents.

For example, suppose a PDF contains records for:

```text
ID 1
ID 2
ID 3
```

A question about:

```text
ID 006
```

may still retrieve the closest existing record.

Without additional validation, the LLM could receive that unrelated record and generate an answer that appears plausible.

The application therefore checks:

```python
best_score < config.RELEVANCE_THRESHOLD
```

If the best result is below the configured threshold, the application does not treat the retrieval as reliable PDF evidence.

---

## 14. Exact ID Validation

The project adds a second protection specifically for identifier-style questions.

The regex recognizes forms such as:

```text
ID 006
ID: 006
ID-006
identifier 006
```

The function:

```python
_exact_id_is_missing(question, results)
```

extracts the requested ID and searches the retrieved text for that same ID.

If the requested ID is not present, the application avoids answering from a different ID's chunk.

This is especially useful for structured-record PDFs where semantic similarity alone may be misleading.

---

## 15. PDF Question vs General Conversation

The chatbot needs to decide whether to use RAG.

The implementation contains:

```python
_looks_like_pdf_question()
```

for document-oriented terminology such as:

```text
pdf
file
document
page
chapter
section
uploaded
id
identifier
record
```

The current decision flow also explicitly handles greetings, memory commands, and the absence of indexed documents before normal RAG retrieval.

The important design principle is:

```text
Do not force every user message through the PDF retriever.
```

A greeting such as `Hi` should not require a PDF chunk.

---

## 16. General Chat Fallback

The `_general_answer()` method sends a separate prompt to Gemini.

It includes:

- The remembered user name, if available.
- Recent conversation history.
- The current user message.

The prompt instructs the model to:

- Respond naturally to greetings.
- Use remembered facts when relevant.
- Avoid pretending information exists in the PDFs.
- Explain politely when document information is unavailable.
- Keep responses concise unless more detail is requested.

This creates a more user-friendly fallback than simply returning:

```text
I cannot find this in the PDF.
```

for every non-document message.

---

## 17. Memory Implementation

The `Memory` class is a small persistent local memory layer.

File:

```text
backend/data/memory.json
```

The default structure is:

```json
{
  "facts": {},
  "history": []
}
```

### Remembering a name

The application checks phrases such as:

```text
My name is Dev
Call me Dev
I am Dev
I'm Dev
```

The extracted name is stored as:

```json
{
  "facts": {
    "name": "Dev"
  }
}
```

The implementation deliberately ignores a few common phrases such as `I am looking` to reduce accidental name extraction.

---

## 18. Direct Memory Question

The chatbot recognizes questions such as:

```text
What's my name?
What is my name?
Do you remember my name?
Who am I?
```

These are answered directly from local memory.

This happens before PDF retrieval, so the user does not need a PDF containing their name.

Example:

```text
User: My name is Dev
Assistant: Nice to meet you, Dev! 😊 I'll remember your name for our future messages.

User: What is my name?
Assistant: Your name is Dev. 😊
```

---

## 19. Conversation History

Every handled user/assistant exchange is stored as a turn:

```json
{
  "role": "user",
  "content": "Hi"
}
```

or:

```json
{
  "role": "assistant",
  "content": "Hello! How can I help?"
}
```

The implementation limits stored history to the most recent 12 turns and uses the most recent 10 turns for general-chat context.

This prevents `memory.json` from growing indefinitely during normal use.

---

## 20. Memory vs RAG Index

These are intentionally separate stores:

```text
PDF knowledge
    ↓
backend/data/index.json

Conversation/user facts
    ↓
backend/data/memory.json
```

A user's name is not embedded into the PDF vector store.

Likewise, PDF chunks are not copied into the memory history.

This separation prevents the assistant from confusing personal conversation context with document evidence.

---

## 21. Complete Question-Answer Decision Tree

The current `RAG.ask()` flow can be summarized as:

```text
User question
     |
     v
Extract possible remembered name
     |
     +---- Is it "What is my name?" / similar?
     |             |
     |             +--> Answer from memory
     |
     +---- Is it a name-setting message?
     |             |
     |             +--> Save name + respond naturally
     |
     +---- Is it casual conversation?
     |             |
     |             +--> General Gemini response
     |
     +---- Is there no PDF index?
     |             |
     |             +--> General Gemini response
     |
     v
Embed question
     |
     v
Cosine similarity search
     |
     v
Top-K results
     |
     +---- No results?
     |       |
     |       +--> General response
     |
     +---- Score below threshold?
     |       |
     |       +--> Friendly fallback
     |
     +---- Requested exact ID missing?
     |       |
     |       +--> Friendly fallback
     |
     v
Build PDF context
     |
     v
Gemini grounded prompt
     |
     v
Answer + source metadata
```

---

## 22. Grounded Prompt Design

For valid PDF questions, the prompt contains:

```text
PDF CONTEXT:
[SOURCE 1] File: ... | Page: ...
...

QUESTION:
...
```

The model is instructed to answer document-related questions only from the supplied PDF context and not invent document facts.

The remembered name is explicitly not treated as evidence for PDF claims.

This is an important RAG design principle:

> **Memory can personalize a response, but memory must not become fake evidence for a document answer.**

---

## 23. Source Citations

The API converts retrieved results into compact source metadata:

```json
{
  "file_name": "example.pdf",
  "page": 3,
  "score": 0.742
}
```

The frontend uses these values to display where the answer came from.

The score is rounded to three decimal places for readability.

---

## 24. Knowledge-Base Statistics

`Store.stats()` returns:

```json
{
  "documents": 2,
  "chunks": 19
}
```

The document count is based on unique filenames in the index.

The chunk count is the number of indexed chunks.

---

## 25. Clearing Data

There are two independent delete operations.

### Clear PDFs

```text
DELETE /api/knowledge-base
```

This clears `index.json` and removes all indexed PDF chunks.

It does **not** clear user memory.

### Clear memory

```text
DELETE /api/memory
```

This resets:

```json
{
  "facts": {},
  "history": []
}
```

It does **not** remove PDFs from the knowledge base.

---

## 26. API Error Handling

The upload endpoint returns HTTP 400 for common user errors, including:

- Non-PDF files.
- Files exceeding the configured size.
- PDF parsing/indexing failures.
- PDFs with no readable text.

The ask endpoint returns HTTP 500 for unexpected model/service failures.

FastAPI/Pydantic validates that the question is:

```text
1–2000 characters
```

---

## 27. Frontend Implementation Responsibilities

`frontend/index.html` is a self-contained client.

The frontend is responsible for:

- Selecting PDF files.
- Drag-and-drop handling.
- Sending multipart upload requests.
- Sending JSON questions.
- Rendering assistant/user messages.
- Displaying source information.
- Loading statistics.
- Clearing PDFs.
- Clearing memory.
- Showing success/error feedback.
- Responsive layout.

It communicates with the backend using relative API paths such as:

```text
/api/upload
/api/ask
/api/stats
/api/memory
/api/knowledge-base
```

Because the frontend is served from the same FastAPI origin, no separate frontend server is necessary.

---

## 28. Startup Implementation

`START_PROJECT.bat` is designed for a non-technical Windows user.

It:

```text
Detect Python
    ↓
Create backend/.venv if missing
    ↓
Install requirements
    ↓
Check backend/.env
    ↓
Ask for API key if needed
    ↓
Start backend window
    ↓
Poll /api/health
    ↓
Open browser
```

The backend launcher runs:

```bat
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The server is intentionally bound to localhost for the local assignment/demo setup.

---

## 29. Why JSON Was Chosen

The project uses JSON instead of a vector database because it keeps the assignment:

- Easy to install.
- Easy to inspect.
- Easy to explain.
- Easy to run on Windows.
- Free of additional infrastructure.

However, a JSON vector store requires scanning the stored vectors during search and is not appropriate for large production datasets.

---

## 30. Production Improvements

For a production deployment, replace the simple components with scalable services.

### Vector storage

Use one of:

- PostgreSQL + pgvector
- Qdrant
- Another managed vector database

### Document storage

Use object storage rather than keeping raw documents only on the application server.

### Processing

Use background jobs for large PDFs and many simultaneous uploads.

### Authentication

Associate every document and memory record with a user/tenant ID.

### Security

Add:

- Authentication.
- Authorization.
- Per-user data isolation.
- Malware scanning.
- Strict MIME/content validation.
- Rate limiting.
- Request limits.
- Secret manager integration.
- HTTPS.
- Audit logs.

### Memory

A production memory system should have explicit:

- User identity.
- Retention policy.
- Deletion semantics.
- Privacy controls.
- Data isolation.
- Potential semantic retrieval of long-term memories.

---

## 31. Testing Checklist

### Application startup

- [ ] Python is detected.
- [ ] Virtual environment is created.
- [ ] Requirements install successfully.
- [ ] `.env` is detected.
- [ ] `/api/health` returns status `ok`.
- [ ] Browser opens on port 8000.

### PDF upload

- [ ] Valid PDF uploads.
- [ ] Multiple PDFs upload.
- [ ] Non-PDF is rejected.
- [ ] Oversized PDF is rejected.
- [ ] Empty/image-only PDF produces a helpful OCR message.
- [ ] Stats update after indexing.

### RAG

- [ ] Known ID returns the correct record.
- [ ] Known content returns an answer.
- [ ] Source filename is displayed.
- [ ] Source page is displayed.
- [ ] Unrelated question does not receive an unrelated PDF answer.
- [ ] Missing ID such as `ID 006` is handled safely.
- [ ] Low-similarity questions use fallback behavior.

### Memory

- [ ] `My name is Dev` stores the name.
- [ ] `What is my name?` returns `Dev`.
- [ ] Memory survives backend restart.
- [ ] Recent conversation is available to general chat.
- [ ] Clear memory removes name/history.
- [ ] Clear PDFs does not remove memory.

### UI

- [ ] Colorful UI loads in normal Chrome.
- [ ] Hard refresh loads the latest frontend.
- [ ] Incognito and normal windows show the same current UI after cache refresh.
- [ ] Upload feedback is visible.
- [ ] Chat input works with Enter.
- [ ] Mobile/responsive layout remains usable.

---

## 32. Example End-to-End Session

Assume the uploaded PDF contains:

```text
ID 1 — Alice
ID 2 — Bob
ID 3 — Charlie
```

### Step 1 — Personal memory

```text
User: My name is Dev
```

The name is stored in `memory.json`.

### Step 2 — General chat

```text
User: Hi
```

The message is handled as general conversation rather than a PDF retrieval request.

### Step 3 — Memory retrieval

```text
User: What is my name?
```

The application directly reads:

```text
facts.name = Dev
```

and returns the remembered name.

### Step 4 — Valid PDF question

```text
User: Who is ID 2?
```

The question is embedded, relevant chunks are retrieved, and Gemini receives the PDF context.

### Step 5 — Missing PDF record

```text
User: Who is ID 006?
```

The requested ID is not found in retrieved context. The application avoids treating ID 1/2/3 as evidence for ID 006 and gives a friendly missing-information response.

---

## 33. Key Design Principles

### Principle 1 — Retrieval is evidence

Similarity results are candidates, not guaranteed facts.

### Principle 2 — Validate structured identifiers

An exact ID requested by the user should be checked against retrieved text.

### Principle 3 — Do not force RAG on casual conversation

Greetings and personal conversation should remain conversational.

### Principle 4 — Memory and document evidence are separate

Remembered personal facts should not be presented as PDF facts.

### Principle 5 — Fail gracefully

When information is missing, explain what happened in natural language instead of exposing a technical error.

### Principle 6 — Keep the demo simple

Local JSON persistence makes the implementation easy to understand and run, while the architecture leaves a clear path toward production infrastructure.
