# EduRAG

EduRAG is an AI-powered personalized teaching assistant built using Retrieval-Augmented Generation (RAG). The system enables students to ask questions from course materials and receive context-aware answers generated using Large Language Models (LLMs).

The project combines semantic retrieval using FAISS embeddings with keyword-based BM25 retrieval to improve response relevance and accuracy for educational content.

---

## Features

### Hybrid Retrieval Pipeline
- Semantic search using FAISS
- Keyword search using BM25
- Combined ranking strategy

### Personalized Learning Experience
- Beginner response mode
- Intermediate response mode
- Advanced response mode

### PDF-Based Knowledge Ingestion
- Upload and process course PDFs
- Automatic text extraction and chunking

### LLM-Powered Responses
- Context-aware answer generation
- Groq API integration for fast inference

### Cloud Storage Integration
- FAISS index persistence using AWS S3
- BM25 and chunk storage support

### Interactive Web Interface
- Streamlit-based UI
- Session management and authentication

---

## Architecture

```text
User Question
      │
      ▼
Hybrid Retriever
 ├── FAISS Semantic Search
 └── BM25 Keyword Search
      │
      ▼
Relevant Context Retrieval
      │
      ▼
Prompt Construction
      │
      ▼
LLM Response Generation (Groq)
      │
      ▼
Personalized AI Tutor Response
