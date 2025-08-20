# importing libraries
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.file_utils import *
from utils.preprocessing import *


# Step 3: Create embeddings
def get_embeddings(texts, embedding_model_name):
    """
    Create embeddings for given texts using HuggingFace embeddings model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype('float32')

    # Normalize embeddings → L2 norm = 1
    faiss.normalize_L2(embeddings)
    return embeddings, embedding_model


# Step 4: Store in FAISS
def store_in_faiss(texts, metadatas, embedding_model, index_path):
    """
    Store texts + metadata in FAISS using the provided embedding model.
    """
    print("Storing in FAISS...")
    faiss_index = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    faiss_index.save_local(index_path)
    print(f"Embeddings stored in FAISS at '{index_path}'")
    return faiss_index


if __name__ == "__main__":
    # Step 1: Extract text
    resumes_data = extract_text()

    # Step 2: Extract key skills from each resume (no chunking)
    skills_texts = []
    metadatas = []
    for name, content in resumes_data.items():
        resume_skills = extract_skills_rake(content)
        resume_skills = remove_stopwords(resume_skills)
        resume_skills = deduplicate_words(resume_skills)
        skills_texts.append(resume_skills)
        metadatas.append({"source": name})

    # Step 3: Get embeddings
    embeddings, embedding_model = get_embeddings(skills_texts, EMBEDDING_MODEL_NAME)

    # Step 4: Store in FAISS
    store_in_faiss(skills_texts, metadatas, embedding_model, FAISS_INDEX_PATH)
