# Med Buddy (xAI) - Advanced RAG Blood Test Assistant

A powerful **Streamlit-based RAG (Retrieval-Augmented Generation)** application that helps users interpret blood test results using **Grok AI** from xAI.

---

## ✨ Overview

**Med Buddy** is an intelligent assistant that allows users to upload PDF blood test reports and ask questions about them. It uses advanced **Retrieval-Augmented Generation (RAG)** techniques with embeddings and vector search to provide accurate, context-aware answers.

The app offers **three response styles** (from simple to professional with treatment suggestions) and full control over document chunking and retrieval.

---

## 🚀 Key Features

- **Three Response Modes**:
  1. Clear and concise (for general public)
  2. Professional and detailed
  3. Professional + Proposed Treatment (if applicable)

- **Advanced Chunking Options**:
  - Full Document
  - Recursive Character Text Splitter (Recommended)
  - Character Text Splitter

- **Custom Chunk Settings**:
  - Adjustable chunk size and overlap

- **Semantic Search (Proper RAG)**:
  - Uses `all-MiniLM-L6-v2` embeddings + **FAISS** vector store
  - Adjustable `Top-K` relevant chunks

- **Chunk Explorer**:
  - View all document chunks
  - Download chunks as `.txt` file

- **Persistent Settings**:
  - "Remember my choice" for response type

- **Strong Medical Disclaimer** in every response

---

## 🛠️ Technologies Used

- **Streamlit** – Web interface
- **Grok-4** (via LangChain) – LLM
- **LangChain** – Orchestration and RAG pipeline
- **FAISS** – Vector database
- **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`)
- **PyPDFLoader** – PDF processing
- **RecursiveCharacterTextSplitter** – Smart chunking

---

## 📋 Requirements

- Python 3.10+
- Grok API Key from [xAI](https://xai.grok.com/)

---

## 🏗️ Installation

1. **Clone the repository** (or download the script):
   ```bash
   git clone <repository-url>
   cd med-buddy