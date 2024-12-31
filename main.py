# main.py

import os
import config
from doc_loader import load_documents_from_file
from rag_pipeline import RAGPipeline

def main():
    # 1. Initialize pipeline
    rag = RAGPipeline(
        model_name=config.MODEL_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        db_path=config.CHROMA_DB_DIR,
        max_length=config.MAX_LENGTH,
        top_p=config.TOP_P
    )

    # 2. If first run, load sample docs
    if not os.path.exists(config.CHROMA_DB_DIR):
        docs = load_documents_from_file(config.DOCS_PATH)
        rag.add_documents(docs)
        print("Sample math documents loaded into ChromaDB.")

    print("Welcome to the Generative AI Math Demo!")
    print("Type a math question or 'exit' to quit.\n")

    while True:
        user_query = input("Enter your math question: ")
        if user_query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        answer = rag.generate_answer(user_query)
        print("\n--- ANSWER ---")
        print(answer)
        print("-------------\n")

if __name__ == "__main__":
    main()
