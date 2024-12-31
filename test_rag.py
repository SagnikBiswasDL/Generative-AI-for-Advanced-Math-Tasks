# tests/test_rag.py

import os
import pytest
import config
from doc_loader import load_documents_from_file
from rag_pipeline import RAGPipeline

def test_document_loading():
    docs = load_documents_from_file(config.DOCS_PATH)
    assert len(docs) > 0, "sample_docs.txt should have lines of math text"

def test_rag_pipeline_initialization():
    rag = RAGPipeline(
        model_name=config.MODEL_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        db_path=config.CHROMA_DB_DIR
    )
    assert rag is not None, "RAGPipeline should initialize properly"

@pytest.mark.skip(reason="Requires GPU or sufficient CPU resources for a real test run")
def test_end_to_end_generation():
    """
    A more complete test that:
    1) Loads docs
    2) Adds them to vector store
    3) Generates an answer
    """
    rag = RAGPipeline(
        model_name=config.MODEL_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        db_path=config.CHROMA_DB_DIR
    )

    if not os.path.exists(config.CHROMA_DB_DIR):
        docs = load_documents_from_file(config.DOCS_PATH)
        rag.add_documents(docs)

    question = "What is the derivative of x^3?"
    answer = rag.generate_answer(question)
    assert "3*x^2" in answer or "3x^2" in answer, "Answer should mention 3*x^2"
