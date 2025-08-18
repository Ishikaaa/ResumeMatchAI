# importing libraries
import os
import docx
import re
from collections import OrderedDict
from config.config_file import *
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

RESUME_STOPWORDS = [
    "implemented", "implementations", "developed", "worked", "working",
    "experience", "responsible", "handled", "performed", "participated", "involved"
]

def deduplicate_words(text):
    """
    Remove repeated words, keep only unique words.
    """
    words = text.split()
    seen = set()
    unique_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    return " ".join(unique_words)

def remove_resume_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in RESUME_STOPWORDS]
    return " ".join(filtered)

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
# def preprocess_text(text):
#     text = text.lower()  # lowercase
#     text = re.sub(r'\s+', ' ', text)  # remove extra spaces
#     # text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
#     return text.strip()
def preprocess_text(text):
    text = text.strip()  # remove leading/trailing spaces
    text = text.lower()   # lowercase
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = remove_resume_stopwords(text)
    return text


# Step 6: Load FAISS index
def load_faiss(faiss_path=faiss_path):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    print(db.index.metric_type)
    print(db.index)
    return db


# Step 7: Retrieve top-5 unique resumes
# def retrieve_top_resumes(db, query, threshold=0.7):
#     # results = db.similarity_search(query, k=20)  # retrieve more
#     results = db.similarity_search_with_score(query, k=20)
#
#     unique_resumes = OrderedDict()
#     for r, score in results:
#         print(r.metadata["source"], score)
#         if score < threshold:  # skip low similarity resumes
#             continue
#         if r.metadata["source"] not in unique_resumes:
#             # store {resume_name: resume_text}
#             unique_resumes[r.metadata["source"]] = r.page_content
#         if len(unique_resumes) == 5:
#             break
#
#     if not unique_resumes:
#         print("No resumes passed the similarity threshold.")
#     return unique_resumes

# Step Y: Retrieve top resumes using JD chunks
def retrieve_top_resumes(db, jd_text, threshold=0.7, k=20):
    """
    Retrieve top-5 unique resumes by comparing JD chunks with FAISS.

    Args:
        db: FAISS vector store.
        jd_text (str): Raw job description.
        threshold (float): Minimum similarity score.
        k (int): Number of results per chunk.

    Returns:
        OrderedDict: {resume_name: resume_text}
    """
    jd_chunks = chunk_job_description(jd_text)
    unique_resumes = OrderedDict()

    for chunk in jd_chunks:
        chunk = deduplicate_words(chunk)
        print(chunk)
        results = db.similarity_search_with_score(chunk, k=k)
        for r, score in results:
            print(r.metadata["source"], score, r.page_content)
            print("***********************")
            if score < threshold:
                continue
            if r.metadata["source"] not in unique_resumes:
                unique_resumes[r.metadata["source"]] = r.page_content
            if len(unique_resumes) == 5:  # stop when we have top 5
                break
        if len(unique_resumes) == 5:
            break

    if not unique_resumes:
        print("No resumes passed the similarity threshold.")
    return unique_resumes


def evaluate():
    # 1. LLM-based Relevance Scoring
    # 2. Keyword Overlap - Extract skills, tools, and qualifications from JD and resume.
    #                      Compute overlap ratio: #matching_keywords / #keywords_in_JD.
    #                      Can serve as a rough automated metric for ranking resumes.
    pass

if __name__ == "__main__":
    # Steps 1–5
    filepath = get_job_description_file()
    raw_text = extract_text_from_docx(filepath)
    clean_text = preprocess_text(raw_text)

    # Step 6
    db = load_faiss()

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Step 7
    top_resumes = retrieve_top_resumes(db, clean_text)

    print("\nTop 5 matching resumes:\n")
    for resume in top_resumes:
        print(resume)
