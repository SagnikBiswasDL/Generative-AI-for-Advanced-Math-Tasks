# doc_loader.py

def load_documents_from_file(file_path):
    """
    Reads a text file, splits each line into its own 'document', 
    and returns them as a list of strings.
    """
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(line)
    return docs
