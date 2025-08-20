from collections import OrderedDict

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.evaluation import *
from utils.generating_reasons import *


def get_job_description_file(filename):
    filepath = os.path.join(JOB_DESCRIPTION_FOLDER, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in JobDescription folder.")
    if not filename.lower().endswith(".docx"):
        raise ValueError("Only .docx files supported.")
    return filepath


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def load_faiss(embedding_model, faiss_path=FAISS_INDEX_PATH):
    # embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

    # Normalize vectors for cosine similarity
    vectors = np.array([db.index.reconstruct(i) for i in range(db.index.ntotal)], dtype='float32')
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vectors
    index.add(vectors)
    db.index = index

    return db


def load_embedding_model():
    """
    Load and return the HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def chunk_job_description(jd_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(jd_text)


def retrieve_top_resumes(db, jd_text, embedding_model, threshold=resume_match_threshold, k=top_k_results):
    """
    Retrieve top resumes matching the Job Description using FAISS search.
    """
    jd_chunks = chunk_job_description(jd_text)
    unique_resumes = OrderedDict()

    for chunk in jd_chunks:
        # Embed chunk
        chunk_emb = embedding_model.embed_query(chunk)
        chunk_emb = np.array(chunk_emb, dtype='float32').reshape(1, -1)
        chunk_emb /= np.linalg.norm(chunk_emb)

        # Search in FAISS
        D, I = db.index.search(chunk_emb, k)
        cosine_scores = 1 - 0.5 * D[0]

        for idx, score in zip(I[0], cosine_scores):
            r = db.docstore._dict[db.index_to_docstore_id[idx]]

            if score < threshold:
                continue

            if r.metadata["source"] not in unique_resumes:
                # unique_resumes[r.metadata["source"]] = r.page_content
                unique_resumes[r.metadata["source"]] = {
                    "content": r.page_content,
                    "score": score
                }

            if len(unique_resumes) == no_unique_resumes:
                break

        if len(unique_resumes) == no_unique_resumes:
            break
    print("Retrieved top 5 resumes successfully..")

    return unique_resumes


def rank_and_explain(db, jd_name, jd_text, embedding_model):
    """
    Retrieve top resumes, generate hiring reasons, and return ranked results.
    """
    top_resumes = retrieve_top_resumes(db, jd_text, embedding_model)
    results = []

    for resume_name, resume_info in top_resumes.items():
        reason = generate_hiring_reason(jd_name, resume_name)
        results.append({
            "resume_name": resume_name,
            "score": resume_info["score"],
            "hiring_reason": reason
        })

    return results


if __name__ == "__main__":
    # Step 1: Get Job Description
    jd_name = input("Please provide Job Description file name (in .docx format): ")
    filepath = get_job_description_file(jd_name)
    raw_text = extract_text_from_docx(filepath)

    # Step 2: Preprocess text
    clean_text = preprocess_text(raw_text)
    keywords_text = extract_keywords(clean_text)

    # Step 3: Load embedding model + FAISS database
    embedding_model = load_embedding_model()
    db = load_faiss(embedding_model)

    # Step 4: Rank resumes & explain hiring reasons
    results = rank_and_explain(db, jd_name, keywords_text, embedding_model)

    # Step 5: Print top resumes (sorted already by score)
    print("\n===== Ranked Resumes (Initial Ranking) =====\n")
    print(f"{'Resume Name':<30} {'Score':<10} {'Hiring Reason'}")
    print("-" * 70)

    for resume in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{resume['resume_name']:<30} {resume['score']:<10.2f} {resume['hiring_reason']}")

    # Step 6: Final evaluation using LLM judge
    evaluation = llm_judge(jd_name, results)

    print("\n===== Final Evaluation (LLM Judge) =====\n")
    print(f"{'Resume Name':<30} {'LLM Score'}")
    print("-" * 45)

    for res in evaluation:
        print(f"{res['resume_name']:<30} {res['llm_score']:.2f}")
