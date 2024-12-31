# Generative AI for Advanced Math

A more "serious" codebase demoing a **Retrieval-Augmented Generation** (RAG) pipeline for math-related queries.  
It uses:

- **LangChain** for embeddings and vector store integration.
- **ChromaDB** as a local storage for math reference documents.
- **Hugging Face Transformers** for text generation.
- **Docker** for containerization (optional).

## Features

- **Context-Aware Math Reasoning**: The pipeline searches a local store of math references before generating answers.
- **Configurable**: Settings (model name, DB path) live in `config.py`, making it easy to swap models or tweak parameters.
- **Separation of Concerns**: Minimal modules (`doc_loader.py`, `rag_pipeline.py`, `main.py`) keep code organized.
- **Testing**: A small test suite in `tests/test_rag.py` verifies basic functionality.

## Quick Start

1. **Clone the repo**:
   ```bash
   git clone https://github.com/YourUsername/gen-ai-math.git
   cd gen-ai-math
