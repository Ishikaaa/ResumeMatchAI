import os
import re

import docx
import google.generativeai as genai
from dotenv import load_dotenv

from config.config_file import *

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
# Configure Gemini API
genai.configure(api_key=gemini_api_key)
# Load Gemini model
gemini_model = genai.GenerativeModel(EVALUATION_MODEL_NAME)


def read_docx(file_path):
    """Read text from a .docx file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])


def llm_judge(jd_filename, top_resumes):
    """
    Evaluate top candidate resumes against a job description using LLM as judge
    """
    jd_path = os.path.join(JOB_DESCRIPTION_FOLDER, jd_filename)
    jd_text = read_docx(jd_path)

    results = []

    for r in top_resumes:
        resume_name = r.get("resume_name")
        resume_path = os.path.join(RESUME_FOLDER, resume_name)

        if not os.path.exists(resume_path):
            print(f"Resume not found: {resume_path}")
            continue

        resume_text = read_docx(resume_path)

        prompt = f"""
        You are a technical hiring expert. 
        Evaluate how well the candidate's resume matches the technical skills required in the job description.

        Instructions:
        1. Extract all technical skills, programming languages, tools, frameworks, and platforms from the Job Description.  
        2. Check if the candidate's resume demonstrates these skills — either explicitly (same keyword) or implicitly (closely related / equivalent skills).  
        3. Assign a score between 0 and 1:
           - 1 = Resume covers all required technical skills.
           - 0 = Resume has none of the required technical skills.
           - Values in between should reflect the proportion of skills covered, with partial matches and related skills considered.

        Job Description:
        {jd_text}

        Resume:
        {resume_text}

        Return ONLY the numeric score as a decimal between 0 and 1. Do not include any explanation, reasoning, or extra text.
        """

        try:
            response = gemini_model.generate_content(prompt)
            llm_response = response.text.strip()

            # Extract the first valid float using regex
            # match = re.search(r"\d*\.?\d+", llm_response)
            match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", llm_response)
            if match:
                score = float(match.group())
                score = min(score, 1.0)  # cap at 1.0
            else:
                score = 0.0
        except Exception as e:
            print(f"Error with Gemini for {resume_name}: {e}")
            score = None

        results.append({
            "resume_name": resume_name,
            "llm_score": score
        })

    results = sorted(results, key=lambda x: x["llm_score"], reverse=True)
    return results

if __name__ == "__main__":
    jd_filename = "JD_AI.docx"
    top_resumes = [{"resume_name": "Candidate Ishika.docx"},
                   {"resume_name": "Candidate 1.docx"},
                   {"resume_name": "Candidate_200.docx"}]

    ans = llm_judge(jd_filename, top_resumes)
    print(ans)
