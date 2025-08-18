import os
import docx
import faiss
from collections import OrderedDict
import numpy as np
from config.config_file import *
from utils.preprocessing import *
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer


# ------------------ Step 1: JD file ------------------
def get_job_description_file():
    filename = "JD_AI.docx"  # TODO: replace with input if needed
    filepath = os.path.join(jobdescription_folder_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in JobDescription folder.")
    if not filename.lower().endswith(".docx"):
        raise ValueError("Only .docx files supported.")
    return filepath

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

# ------------------ Step 2: Load FAISS ------------------
def load_faiss(embedding_model, faiss_path=faiss_path):
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

# ------------------ Step 3: Chunk JD ------------------
def chunk_job_description(jd_text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(jd_text)

# ------------------ Step 4: Retrieve top resumes ------------------
def retrieve_top_resumes(db, jd_text, embedding_model, threshold=0.7, k=20):
    jd_chunks = chunk_job_description(jd_text)
    unique_resumes = OrderedDict()

    for chunk in jd_chunks:
        print(f"## chunk: {chunk}")
        chunk_emb = embedding_model.embed_query(chunk)
        chunk_emb = np.array(chunk_emb, dtype='float32').reshape(1, -1)
        chunk_emb /= np.linalg.norm(chunk_emb)

        D, I = db.index.search(chunk_emb, k)
        cosine_scores = 1 - 0.5 * D[0]

        for idx, score in zip(I[0], cosine_scores):
            r = db.docstore._dict[db.index_to_docstore_id[idx]]
            print(f'@@ name: {r.metadata["source"]}')
            print(f'@@ score: {score}')
            print(f'@@ content: {r.page_content}')
            print("*********************************************")
            if score < threshold:
                continue
            if r.metadata["source"] not in unique_resumes:
                unique_resumes[r.metadata["source"]] = r.page_content
            if len(unique_resumes) == 5:
                break
        if len(unique_resumes) == 5:
            break

    return unique_resumes


# ------------------ Step 5: Summarization ------------------
def summarize_text(text, llm_model_name="google/flan-t5-base", max_input_tokens=400, max_output_tokens=120):
    summarizer = pipeline("text2text-generation", model=llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_input_tokens] for i in range(0, len(tokens), max_input_tokens)]
    summaries = []

    for chunk_tokens in chunks:
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        prompt = f"Summarize the following in 2-3 concise lines:\n{chunk_text}"
        summary = summarizer(prompt, max_new_tokens=max_output_tokens, do_sample=False)[0]['generated_text']
        summaries.append(summary.strip())

    combined_summary = " ".join(summaries)
    final_prompt = f"Refine the following into 2-3 concise lines:\n{combined_summary}"
    final_summary = summarizer(final_prompt, max_new_tokens=max_output_tokens, do_sample=False)[0]['generated_text']
    return final_summary.strip()

def truncate_resume_by_tokens(resume_text, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokens = tokenizer.encode(resume_text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

def summarize_resume(resume_text, llm_model, max_tokens=512):
    truncated_text = truncate_resume_by_tokens(resume_text, max_tokens)
    prompt = f"Candidate Resume: {truncated_text}\nSummarize key skills & experience in 2-3 lines."
    summary = llm_model(prompt, max_length=150, do_sample=False)[0]['generated_text']
    return summary.strip()

# ------------------ Step 6: Hiring reasons ------------------
def generate_hiring_reasons(unique_resumes, job_description):
    qa_model = pipeline("text2text-generation", model=llm_model)
    jd_summary = summarize_text(job_description)
    hire_reasons = {}

    for candidate, resume_text in unique_resumes.items():
        resume_summary = summarize_resume(resume_text, qa_model)
        prompt = f"Job Description Summary: {jd_summary}\nCandidate Resume Summary: {resume_summary}\nWhy hire this candidate? Answer in 2-3 lines."
        response = qa_model(prompt, max_new_tokens=80, do_sample=False)[0]['generated_text']
        hire_reasons[candidate] = response.strip()

    return hire_reasons

# ------------------ Step 7: Wrapper ------------------
def rank_and_explain(db, jd_text, embedding_model):
    top_resumes = retrieve_top_resumes(db, jd_text, embedding_model)
    return generate_hiring_reasons(top_resumes, jd_text)

# ------------------ Main ------------------
if __name__ == "__main__":
    filepath = get_job_description_file()
    raw_text = extract_text_from_docx(filepath)
    clean_text = preprocess_text(raw_text)
    keywords_text = extract_skills_rake(clean_text)
    keywords_text = remove_stopwords(keywords_text)
    keywords_text = deduplicate_words(keywords_text)

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = load_faiss(embedding_model)

    results = rank_and_explain(db, keywords_text, embedding_model)
    for i, j in results.items():
        print(i, j)

