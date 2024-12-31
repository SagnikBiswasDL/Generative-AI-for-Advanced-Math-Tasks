# rag_pipeline.py

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class RAGPipeline:
    def __init__(self, model_name, embedding_model, db_path, max_length=150, top_p=0.9):
        """
        A simple Retrieval-Augmented Generation pipeline setup.
        """
        # 1. Embeddings
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # 2. Vector store (ChromaDB)
        self.db_path = db_path
        self.vectorstore = Chroma(
            collection_name="math_docs",
            embedding_function=self.embedder,
            persist_directory=db_path
        )
        
        # 3. Text generation model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.llm = HuggingFacePipeline(pipeline=generator)

        self.max_length = max_length
        self.top_p = top_p

    def add_documents(self, docs):
        """
        Add new documents (list of strings) to the Chroma vector store.
        """
        docs_to_add = [{"page_content": doc, "metadata": {}} for doc in docs]
        self.vectorstore.add_documents(docs_to_add)
        self.vectorstore.persist()

    def retrieve_context(self, query, k=2):
        """
        Retrieve top-k relevant docs from the vector store.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        combined_context = "\n".join([res.page_content for res in results])
        return combined_context

    def generate_answer(self, query):
        """
        1) Retrieve relevant context
        2) Concatenate context + question
        3) Use model to generate an answer
        """
        context = self.retrieve_context(query)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        response = self.llm(prompt, max_length=self.max_length, do_sample=True, top_p=self.top_p)
        # The pipeline returns a list of { 'generated_text': '...' }
        answer = response[0]["generated_text"]

        # If you want to strip out prompt boilerplate:
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[-1].strip()
        return answer
