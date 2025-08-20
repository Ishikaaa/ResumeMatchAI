# resume_folder_path = "data//Resumes"  # Folder name containing resumes
# jobdescription_folder_path = "data//JobDescription"
# Path to folder
RESUME_FOLDER = "data/Resumes"
JOB_DESCRIPTION_FOLDER = "data/JobDescription"
FAISS_INDEX_PATH = "resume_faiss_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EVALUATION_MODEL_NAME = "gemini-1.5-flash"
llm_model = "google/flan-t5-base"
# llm_model_name = "google/flan-t5-small"

# hyperparameters
chunk_size = 300
chunk_overlap = 50
resume_match_threshold = 0.7
top_k_results = 20
no_unique_resumes = 5
