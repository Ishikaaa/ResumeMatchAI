# importing libraries
import os
import docx
import re
import faiss
from collections import OrderedDict
from config.config_file import *
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from rake_nltk import Rake
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# Step 1: Take user input for file name
def get_job_description_file():
    while True:
        # filename = input("Enter job description file name (with extension): ").strip()      # TODO - uncomment
        filename = "JD_AI.docx"                                                        # TODO - remove
        filepath = os.path.join(jobdescription_folder_path, filename)

        # Step 2: Check if file exists in JobDescription folder
        if not os.path.exists(filepath):
            print(f"File '{filename}' not found in 'JobDescription' folder. Try again.")
            continue

        # Step 3: Check file extension
        if not filename.lower().endswith(".docx"):
            print("Only .docx format is supported. Please provide a .docx file.")
            continue

        return filepath

# Step 4: Extract text from .docx file
def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Step 5: Preprocess text
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    # text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip()

# Step 6: Load FAISS index
def load_faiss(faiss_path=faiss_path):
    # embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)       # TODO - write another function for this
    # db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    # print("db.index", type(db.index))
    # return db

    # embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    #
    # # Normalize all stored embeddings for cosine similarity
    # faiss.normalize_L2(db.index.reconstruct_n(0, db.index.ntotal))
    #
    # print("db.index", type(db.index))
    # return db

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

    # Normalize all stored embeddings for cosine similarity
    # Convert FAISS index to numpy array, normalize, and recreate index
    vectors = np.array([db.index.reconstruct(i) for i in range(db.index.ntotal)], dtype='float32')
    faiss.normalize_L2(vectors)  # L2 normalize vectors

    # Rebuild FAISS index with normalized vectors
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = inner product = cosine similarity on normalized vectors
    index.add(vectors)
    db.index = index

    print("db.index", type(db.index))
    return db

def chunk_job_description(jd_text, chunk_size=300, chunk_overlap=50):
    """
    Split the Job Description text into chunks suitable for embedding and retrieval.

    Args:
        jd_text (str): Raw job description text.
        chunk_size (int): Max number of characters per chunk.
        chunk_overlap (int): Overlap between chunks to maintain context.

    Returns:
        List[str]: List of JD text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    jd_chunks = text_splitter.split_text(jd_text)
    return jd_chunks

# def retrieve_top_resumes(db, jd_text, threshold=0.7, k=20):
#     jd_chunks = chunk_job_description(jd_text)
#     unique_resumes = OrderedDict()
#
#     for chunk in jd_chunks:
#         print("chunk:", chunk)
#         # 1️⃣ Embed and normalize the JD chunk
#         chunk_emb = db.embedding_model.embed_query(chunk)
#         chunk_emb = chunk_emb / np.linalg.norm(chunk_emb)
#
#         results = db.similarity_search_with_score(chunk, k=k)
#         for r, score in results:
#             print(r.metadata["source"], score, r.page_content)
#             print("***********************")
#             if score < threshold:
#                 continue
#             if r.metadata["source"] not in unique_resumes:
#                 unique_resumes[r.metadata["source"]] = r.page_content
#             if len(unique_resumes) == 5:  # stop when we have top 5
#                 break
#         if len(unique_resumes) == 5:
#             break
#
#     if not unique_resumes:
#         print("No resumes passed the similarity threshold.")
#     return unique_resumes

def retrieve_top_resumes(db, jd_text, embedding_model, threshold=0.7, k=20):
    jd_chunks = chunk_job_description(jd_text)
    unique_resumes = OrderedDict()

    for chunk in jd_chunks:
        print("chunk:", chunk)

        # Embed & normalize query
        chunk_emb = embedding_model.embed_query(chunk)
        chunk_emb = np.array(chunk_emb, dtype='float32').reshape(1, -1)
        chunk_emb /= np.linalg.norm(chunk_emb)

        # Search in FAISS index
        D, I = db.index.search(chunk_emb, k)
        cosine_scores = 1 - 0.5 * D[0]

        for idx, score in zip(I[0], cosine_scores):
            r = db.docstore._dict[db.index_to_docstore_id[idx]]
            print(r.metadata["source"], score, r.page_content)
            print("***********************")
            if score < threshold:
                continue
            if r.metadata["source"] not in unique_resumes:
                unique_resumes[r.metadata["source"]] = r.page_content
            if len(unique_resumes) == 5:
                break
        if len(unique_resumes) == 5:
            break

    if not unique_resumes:
        print("No resumes passed the similarity threshold.")
    return unique_resumes

def summarize_job_description(jd_text, llm_model_name="google/flan-t5-base",
                              max_input_tokens=400, max_output_tokens=120):
    """
    Summarize a long Job Description into 2-3 concise lines using chunking.
    Removes repetition and focuses on unique skills and responsibilities.
    """
    summarizer = pipeline("text2text-generation", model=llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    tokens = tokenizer.encode(jd_text)
    chunks = [tokens[i:i+max_input_tokens] for i in range(0, len(tokens), max_input_tokens)]

    summaries = []
    for chunk_tokens in chunks:
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        prompt = f"""
        Summarize the following job description in 2-3 concise lines.
        - Highlight ONLY key skills, experience, and responsibilities.
        - Avoid any repetition or copying sentences verbatim.
        - Keep each point unique and concise.

        Job Description Chunk:
        {chunk_text}
        """
        summary = summarizer(prompt, max_new_tokens=max_output_tokens, do_sample=False)[0]['generated_text']
        summaries.append(summary.strip())

    # Combine all chunk summaries and do a final summarization pass
    combined_summary = " ".join(summaries)
    final_prompt = f"""
    Refine the following summary into 2-3 concise lines.
    - Remove all repeated points.
    - Keep only unique key skills and responsibilities.

    Summary:
    {combined_summary}
    """
    final_summary = summarizer(final_prompt, max_new_tokens=max_output_tokens, do_sample=False)[0]['generated_text']
    print("**** final_summary", final_summary)
    return final_summary.strip()

# Function to truncate resume text based on model tokens
def truncate_resume_by_tokens(resume_text, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokens = tokenizer.encode(resume_text, truncation=True, max_length=max_tokens)
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

# Example usage inside your summarize_resume function
def summarize_resume(resume_text, llm_model, max_tokens=512):
    truncated_text = truncate_resume_by_tokens(resume_text, max_tokens=512)

    prompt = f"""
    Candidate Resume: {truncated_text}
    Question: Summarize the key skills, experience, and qualifications of this candidate in 2-3 lines.
    """
    summary = llm_model(prompt, max_length=150, do_sample=False)[0]['generated_text']
    return summary.strip()

# Step 8: Generate 2-line justification using LLM
def generate_hiring_reasons(unique_resumes, job_description):
    # Load free instruction-tuned model
    qa_model = pipeline("text2text-generation", model=llm_model)
    hire_reasons = {}
    jd_summary = summarize_job_description(job_description)

    for candidate, resume_text in unique_resumes.items():
        resume_summary = summarize_resume(resume_text, qa_model)
        # Create direct question prompt
        # prompt = f"""
        # Job Description: {job_description}
        # Candidate Resume: {resume_text[:1000]}  # truncate to avoid token overflow
        # Question: Why should we hire this candidate?
        # Answer in 2-3 lines.
        # """

        prompt = f"""
        Job Description Summary: {jd_summary}
        Candidate Resume Summary: {resume_summary}

        Question: Why should we hire this candidate? Answer in 2-3 lines.
        """

        # response = qa_model(prompt, max_length=80, do_sample=False)[0]['generated_text']
        response = qa_model(prompt, max_new_tokens=80, do_sample=False)[0]['generated_text']
        hire_reasons[candidate] = response.strip()

    return hire_reasons

# Step 9: Final wrapper
def rank_and_explain(db, jd_text, embeddings_model):
    # Step 7
    top_resumes = retrieve_top_resumes(db, jd_text, embeddings_model)

    # Step 8
    top_resumes_reasons = generate_hiring_reasons(top_resumes, jd_text)

    # Step 9 → Return combined result
    return top_resumes_reasons

def extract_skills_rake(jd_text, top_k=20):
    r = Rake()
    r.extract_keywords_from_text(jd_text)
    # phrases = r.get_ranked_phrases()[:top_k]
    phrases = r.get_ranked_phrases()

    # Remove . and ) from phrases
    clean_phrases = [re.sub(r"[.)]", "", phrase) for phrase in phrases]
    jd_keywords_text = " ".join(clean_phrases)
    return jd_keywords_text


if __name__ == "__main__":
    # Steps 1–5
    filepath = get_job_description_file()
    raw_text = extract_text_from_docx(filepath)
    clean_text = preprocess_text(raw_text)
    keywords_text = extract_skills_rake(clean_text, top_k=20)
    # embedding = get_embedding(keywords_text)

    # Step 6
    db = load_faiss()

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # # Step 7
    # top_resumes = retrieve_top_resumes(db, clean_text)

    # print("\nTop 5 matching resumes:\n")
    # for resume in top_resumes:
    #     print(resume)

    # embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 7–9 → Rank and explain in one go
    results = rank_and_explain(db, clean_text, embedding_model)

    # print("\nFinal Top 5 Candidates with Reasons:\n")
    # for candidate, reason in results.items():
    #     print(f"{candidate}: {reason}")
