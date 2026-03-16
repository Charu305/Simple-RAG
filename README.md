# 🤖 Simple RAG — Retrieval-Augmented Generation Pipeline

> **A Retrieval-Augmented Generation (RAG) system** that allows users to ask natural language questions against a private PDF document — combining vector search over document chunks with a Large Language Model (LLM) to generate accurate, grounded answers from a Company HR Policy Handbook.

---

## 📌 Project Overview

Large Language Models (LLMs) are powerful but have two critical limitations for enterprise use:
1. **They don't know your private data** — trained on public internet data, not your internal documents
2. **They hallucinate** — when they don't know something, they confidently make up an answer

**RAG solves both problems.** Instead of relying purely on the LLM's parametric knowledge, RAG first *retrieves* the most relevant chunks from a private knowledge base, then *augments* the LLM's prompt with that retrieved context before generating a response.

This project implements a complete RAG pipeline end-to-end — from PDF ingestion and chunking, to embedding, vector storage, semantic retrieval, and LLM-powered answer generation — using a **Company HR Policy Handbook** as the private knowledge source.

---

## 🎯 Problem Statement

> *Given a PDF document (HR Policy Handbook), build a system where a user can ask any natural language question and receive an accurate, document-grounded answer — without the LLM hallucinating information not present in the document.*

**Real-world applications of RAG:**
- **HR bots** — answer employee questions from policy documents instantly
- **Legal assistants** — query contracts, regulations, and case documents
- **Customer support** — answer FAQs from product manuals or knowledge bases
- **Research tools** — query scientific papers or internal reports
- **Enterprise Q&A** — make internal documentation searchable through natural conversation

---

## 🏗️ RAG Pipeline Architecture

```
PDF Document
(Company HR Policy Handbook.pdf)
            │
            ▼
┌──────────────────────────────────┐
│   1. Document Loading            │
│   Extract raw text from PDF      │
│   (PyPDF2 / pdfplumber)          │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│   2. Text Chunking               │
│   Split into overlapping chunks  │
│   (e.g., 500 tokens, 50 overlap) │
│   Preserves context at boundaries│
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│   3. Embedding Generation        │
│   Each chunk → dense vector      │
│   (Sentence Transformers /       │
│    OpenAI Embeddings)            │
└────────────┬─────────────────────┘
             │  Vectors stored in
             ▼  vector index
┌──────────────────────────────────┐
│   4. Vector Store                │
│   FAISS / ChromaDB               │
│   Enables fast similarity search │
└──────────────────────────────────┘

            At Query Time:
            ─────────────
User Question
      │
      ▼
Embed question → query vector
      │
      ▼
Similarity search → Top-K relevant chunks
      │
      ▼
Augmented Prompt:
"Answer based on this context: {chunks}
Question: {user_question}"
      │
      ▼
LLM (GPT / open-source model)
      │
      ▼
Grounded Answer ✅
```

---

## 🗂️ Project Structure

```
Simple-RAG/
│
├── RAG.ipynb                        # Full RAG pipeline notebook
└── Company HR Policy Handbook.pdf   # Knowledge source document
```

---

## 🔬 Technical Deep Dive

### 1. Document Loading

- Loaded `Company HR Policy Handbook.pdf` using a PDF parsing library.
- Extracted raw text page by page, preserving document structure where possible.
- Handled multi-page documents, headers, footers, and formatting artefacts common in HR policy PDFs.

### 2. Text Chunking

Chunking is one of the most critical decisions in a RAG pipeline:

| Strategy | Description |
|---|---|
| **Fixed-size chunking** | Split text into chunks of N tokens with overlap |
| **Sentence-aware chunking** | Split at sentence boundaries to avoid mid-sentence cuts |
| **Overlap** | Each chunk shares N tokens with the previous — prevents losing context at boundaries |

**Why overlap matters:** If the answer spans two chunks, overlap ensures the relevant context appears fully in at least one retrieved chunk rather than being split across two.

### 3. Embedding Generation

- Each text chunk is converted into a **dense vector embedding** using a sentence transformer model.
- Embeddings capture **semantic meaning** — chunks about "annual leave policy" and "vacation days entitlement" get similar vectors even with different words.
- This enables **semantic search** — finding relevant chunks by meaning, not just keyword matching.

**Models used:**
- `sentence-transformers/all-MiniLM-L6-v2` — lightweight, fast, strong semantic similarity
- OpenAI `text-embedding-ada-002` (if API key available) — higher quality embeddings

### 4. Vector Store (FAISS / ChromaDB)

- All chunk embeddings are stored in a **vector index** (FAISS or ChromaDB).
- At query time, the user's question is embedded using the same model.
- A **cosine similarity / nearest-neighbour search** retrieves the Top-K most semantically similar chunks.
- This retrieval step is what grounds the LLM's response in actual document content.

### 5. Augmented Prompt Construction

Retrieved chunks are injected into the LLM prompt as context:

```
System: You are a helpful HR assistant. Answer questions using ONLY the provided context.
        If the answer is not in the context, say "I don't know based on the provided document."

Context:
{retrieved_chunk_1}
{retrieved_chunk_2}
{retrieved_chunk_3}

Question: {user_question}

Answer:
```

This prompt design:
- **Grounds** the LLM in retrieved document content
- **Prevents hallucination** by explicitly instructing it to stay within context
- **Handles out-of-scope questions** gracefully with the "I don't know" instruction

### 6. Answer Generation

- The augmented prompt is sent to the LLM.
- The LLM synthesises the retrieved context into a coherent, natural language answer.
- The answer is traceable back to specific chunks in the source document.

---

## 📊 Why RAG over Fine-Tuning?

A common interview question: *"Why use RAG instead of fine-tuning the LLM on your documents?"*

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge updates** | Update the vector store — no retraining needed | Requires full retraining |
| **Cost** | Low — only inference costs | High — GPU training costs |
| **Hallucination control** | Strong — grounded in retrieved context | Weaker — model may still confabulate |
| **Transparency** | High — can show source chunks used | Low — knowledge baked into weights |
| **Data privacy** | Document stays local | Data sent to training pipeline |
| **Best for** | Dynamic, frequently updated documents | Learning new tasks or writing styles |

**Verdict:** For enterprise documents like HR handbooks, legal policies, or product manuals that change regularly — RAG is almost always the right choice over fine-tuning.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| PDF Parsing | PyPDF2 / pdfplumber |
| Text Chunking | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings | Sentence Transformers / OpenAI Embeddings |
| Vector Store | FAISS / ChromaDB |
| LLM | OpenAI GPT / open-source LLM (via API) |
| Orchestration | LangChain |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Charu305/Simple-RAG.git
cd Simple-RAG

# 2. Install dependencies
pip install langchain openai faiss-cpu sentence-transformers pypdf2 jupyter

# 3. Set your OpenAI API key (if using GPT)
export OPENAI_API_KEY="your-api-key-here"

# 4. Launch the notebook
jupyter notebook RAG.ipynb
```

---

## 💬 Example Interactions

```
User:  What is the leave policy for new employees?
Bot:   According to the HR Policy Handbook, new employees are entitled to...
       [grounded in retrieved document chunk]

User:  How many sick days are allowed per year?
Bot:   The policy states that employees are entitled to X sick days per year...
       [grounded in retrieved document chunk]

User:  What is the company's stock price?
Bot:   I don't know based on the provided document.
       [correct — out-of-scope question handled gracefully]
```

---

## 💡 Key Learnings & Takeaways

- **RAG is the most practical GenAI architecture for enterprise** — it combines the language capability of LLMs with the factual grounding of your own documents, without the cost and complexity of fine-tuning.
- **Chunking strategy significantly impacts retrieval quality** — chunks too large dilute relevance; chunks too small lose context. Overlap at chunk boundaries is essential for questions that span sections.
- **Embeddings enable semantic search, not keyword search** — a user asking "how many vacation days do I get?" will retrieve chunks about "annual leave entitlement" even without those exact words in the query.
- **Prompt engineering controls hallucination** — explicitly instructing the LLM to answer *only* from context and to say "I don't know" when the answer isn't present is what keeps RAG honest.
- **The vector store is the memory** — FAISS and ChromaDB are purpose-built for fast similarity search over millions of vectors, making retrieval nearly instant even on large document collections.

---

## 🔮 Potential Enhancements

- **Hybrid search** — combine dense vector search with keyword (BM25) search for better recall on precise terms (policy codes, section numbers)
- **Re-ranking** — add a cross-encoder re-ranker to re-score retrieved chunks before passing to the LLM for higher precision
- **Multi-document RAG** — extend to query across multiple PDFs (entire policy library, product documentation)
- **Evaluation with RAGAS** — measure Faithfulness, Answer Relevance, and Context Precision systematically
- **Streaming responses** — stream LLM output token-by-token for a more responsive chat experience

---

## 👩‍💻 Author

**Charunya**
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

This project is developed for educational and research purposes.
