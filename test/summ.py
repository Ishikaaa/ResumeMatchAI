# # # # Working
# # #
# # # # from transformers import pipeline
# # # #
# # # # # Use a seq2seq model like Flan-T5
# # # # summarizer = pipeline(
# # # #     "text2text-generation",
# # # #     model="google/flan-t5-base",
# # # #     tokenizer="google/flan-t5-base"
# # # # )
# # # #
# # # # def resume_summary(resume_text):
# # # #     prompt = f"""
# # # # check grammer as this I am giving to LLM model as prompt - You are an expert recruiter. Extract relevant skills, work experience, Education, Certifications, Projects details.
# # # # Make sure to not miss even 1 skill from resume as this is very important for candiadate and for company
# # # #
# # # # Resume Text:
# # # # {resume_text}
# # # # """
# # # #     result = summarizer(prompt, max_new_tokens=1024, temperature=0)
# # # #     return result[0]["generated_text"]
# # # #
# # # # # Example usage
# # # # resume_text = """
# # # # Ishika Garg
# # # # AI/ML Developer
# # # # Ishika Garg      Ishika       Ishika      Ishika-medium
# # # # +91-8699584599      gargishika1998@gmail.com
# # # # EXPERIENCE
# # # #
# # # # Genpact                                                                                                                               Aug 2023 - Present
# # # # AI/ML Developer
# # # # Built a framework integrating AIOps features, including noise reduction, self-resolution, and chatbot support using Linear SVC, BERT, and RASA, enabling resolution of 70% of tickets within 20–40 seconds of submission.
# # # # Developed an end-to-end PDF extraction solution by experimenting with multiple LLMs, including fine-tuning and prompt engineering, leveraging RAG and FAISS vector database to automate user SOP generation, significantly reducing manual effort.
# # # # Built end-to-end automation solutions by understanding client requirements, designing workflows, implementing solutions, and delivering client demos. Reduced manual processing time from 30 minutes to 30 seconds per ticket.
# # # # Tech Stack - Python, scikit-learn, PDFPlumber, RASA, Linear SVC, BERT, LLM, Instruct-XL, RAG, FAISS
# # # # Imagination Technologies                                                                                      Sep 2022 - June 2023
# # # # Software Developer
# # # # Developed a model evaluation tool using custom preprocessing, inference, and postprocessing plugins, enabling the evaluation of 50+ neural networks and reducing evaluation time by 60%, while standardizing accuracy measurement across teams.
# # # # Engaged classification and LSTM neural networks using diverse inputs like images and MNIST datasets, enhancing model versatility and performance.
# # # # Tech Stack - Python, TensorFlow, NumPy, scikit-learn, git
# # # # Infosys                                                                                                                              Oct 2020 - Aug 2022
# # # # Senior Systems Engineer
# # # # Developed automated scripts for an end-to-end solution, from document collection (via web scraping, email, and PACER) to preprocessing and uploading to a user-friendly single website using core Python, cutting manual efforts by 70%.
# # # # Tech Stack - Python, BeautifulSoup, Selenium, Pandas, git
# # # # TECHNICAL SKILLS
# # # #
# # # # Technical Skills: Python, Machine Learning (algorithms - Linear Regression, Logistic Regression, Linear SVM, Random Forest), Deep Learning (NLP, RNN, LLMs)
# # # # Libraries & Tools: Pandas, NumPy,  Transformers, Hugging Face, NLTK, RASA, Langchain, RAG, Streamlit, JIRA, PyTorch
# # # # Databases & Cloud: MySQL, Vector Databases(FAISS), AWS
# # # # Other: MLOps, Github, Jenkins, CI/CD, Linux
# # # # EDUCATION
# # # # Indian Institute of Science, Bangalore                                                                                                  2024 - Present
# # # # Post Graduate Diploma (Deep Learning)
# # # # Punjabi University, Patiala                                                                                                                             2016 - 2020
# # # # Bachelor of Technology (CSE)
# # # # PROJECTS
# # # #
# # # # Medichat | Code
# # # # Python, LangChain, Streamlit, Hugging Face, FAISS, pypdf
# # # # •  Developing an AI-powered medical chatbot using LLMs, enhanced with prompt engineering to deliver context-aware responses. Exploring model fine-tuning for improved medical accuracy.
# # # #
# # # # Classification of Nepal Earthquake Tweets | Code
# # # # Pandas, numpy, matplotlib, sklearn, TfidfVectorizer, Logistic regression, accuracy_score
# # # # •  Developed a tool to classify tweets into relevant or irrelevant categories during crises like the Nepal Earthquake by preprocessing data(handling missing values, feature extraction) and applying logistic regression for effective categorization.
# # # # CERTIFICATIONS
# # # #
# # # # •  Machine Learning with Python – Level 1, issued by IBM
# # # # •  Data Science Foundations – Level 1, issued by IBM
# # # # •  Problem Solving(Basic and intermediate) certificate by HackerRank
# # # # •  Python certification by HackerRank
# # # #
# # # # """
# # # # print(resume_summary(resume_text))
# # #
# # #
# # #
# # # from transformers import pipeline
# # #
# # # # Load an instruction-tuned LLM (you can replace with any HuggingFace model)
# # # # pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device=0)
# # # pipe = pipeline("text2text-generation", model="google/flan-t5-base")
# # #
# # # def clean_resume_text(text):
# # #     """
# # #     Sends resume text to LLM and asks it to ONLY remove irrelevant words.
# # #     """
# # #     prompt = f"""
# # #     You are a resume text cleaner.
# # #     Your ONLY job is to remove irrelevant words like "built", "developed", "by", "understanding", "created", "made", "Exploring", "Developing", etc.
# # #     Do not add or rephrase. Only delete words which are not important to select a resume.
# # #
# # #     Example:
# # #     Input: "Built end-to-end automation solutions by understanding client requirements, designing workflows, implementing solutions, and delivering client demos. Reduced manual processing time from 30 minutes to 30 seconds per ticket. Tech Stack - Python, scikit-learn, PDFPlumber, RASA, Linear SVC, BERT, LLM, Instruct-XL, RAG, FAISS"
# # #
# # #     Output: "end-to-end automation solutions understanding client requirements, designing workflows, implementing solutions, delivering client demos. Python, scikit-learn, PDFPlumber, RASA, Linear SVC, BERT, LLM, Instruct-XL, RAG, FAISS"
# # #
# # #     Now clean this text:
# # #     Input: {text}
# # #     Output:
# # #     """
# # #
# # #     # response = pipe(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
# # #     pipe = pipeline("text2text-generation", model="google/flan-t5-base")
# # #     response = pipe(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
# # #
# # #     # Post-process: extract only cleaned part after "Cleaned text:"
# # #     if "Cleaned text:" in response:
# # #         response = response.split("Cleaned text:")[-1].strip()
# # #     print(response)
# # #
# # #     return response
# # #
# # #
# # # if __name__ == "__main__":
# # #     resume_chunk = """
# # # Senior Systems Engineer
# # # Developed automated scripts for an end-to-end solution, from document collection (via web scraping, email, and PACER) to preprocessing and uploading to a user-friendly single website using core Python, cutting manual efforts by 70%.
# # # Tech Stack - Python, BeautifulSoup, Selenium, Pandas, git
# # # TECHNICAL SKILLS
# # #
# # # Technical Skills: Python, Machine Learning (algorithms - Linear Regression, Logistic Regression, Linear SVM, Random Forest), Deep Learning (NLP, RNN, LLMs)
# # # Libraries & Tools: Pandas, NumPy,  Transformers, Hugging Face, NLTK, RASA, Langchain, RAG, Streamlit, JIRA, PyTorch
# # # Databases & Cloud: MySQL, Vector Databases(FAISS), AWS
# # # Other: MLOps, Github, Jenkins, CI/CD, Linux
# # # EDUCATION
# # # Indian Institute of Science, Bangalore                                                                                                  2024 - Present
# # # Post Graduate Diploma (Deep Learning)
# # # Punjabi University, Patiala                                                                                                                             2016 - 2020
# # # Bachelor of Technology (CSE)
# # # PROJECTS
# # #
# # # Medichat | Code
# # # Python, LangChain, Streamlit, Hugging Face, FAISS, pypdf
# # # •  Developing an AI-powered medical chatbot using LLMs, enhanced with prompt engineering to deliver context-aware responses. Exploring model fine-tuning for improved medical accuracy.
# # #
# # # Classification of Nepal Earthquake Tweets | Code
# # # Pandas, numpy, matplotlib, sklearn, TfidfVectorizer, Logistic regression, accuracy_score
# # # •  Developed a tool to classify tweets into relevant or irrelevant categories during crises like the Nepal Earthquake by preprocessing data(handling missing values, feature extraction) and applying logistic regression for effective categorization.
# # # CERTIFICATIONS
# # #
# # # •  Machine Learning with Python – Level 1, issued by IBM
# # # •  Data Science Foundations – Level 1, issued by IBM
# # # •  Problem Solving(Basic and intermediate) certificate by HackerRank
# # # •  Python certification by HackerRank                                                                                                                            Oct 2020 - Aug 2022
# # #
# # # """
# # #
# # #
# # #
# # #
# # #     cleaned_text = clean_resume_text(resume_chunk)
# # #     # print("Original:", resume_chunk)
# # #     # print("Cleaned :", cleaned_text)
# #
# #
# #
# # from transformers import pipeline
# #
# # # Load summarization/text2text pipeline
# # pipe = pipeline("text2text-generation", model="google/flan-t5-base")
# # # pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", max_new_tokens=512, do_sample=False)
# #
# # def clean_resume_with_llm(text):
# #     prompt = f"""You are a STRICT deletion-only editor.
# #
# #     RULES:
# #     - You may ONLY delete the following filler words/particles: built, build, developed, develop, created, create, made, by, the, a, an.
# #     - Do NOT add, rephrase, reorder, or paraphrase anything.
# #     - Keep ALL technical terms, skills, entities, numbers, and domain phrases exactly as-is.
# #     - Output MUST be wrapped EXACTLY as:
# #     <cleaned>...only the cleaned text here...</cleaned>
# #     - Do not include any other words outside the tags.
# #
# #     EXAMPLES
# #     Input: Built end-to-end automation using Python, RASA, and FAISS by the team.
# #     Output:
# #     <cleaned>end-to-end automation using Python, RASA, and FAISS team.</cleaned>
# #
# #     Input: Developed model evaluation tools and created pipelines by the data team.
# #     Output:
# #     <cleaned>model evaluation tools and pipelines data team.</cleaned>
# #
# #     Now process this input:
# #     {text}
# #     Output:
# #     """
# #     result = pipe(prompt, max_new_tokens=512)[0]['generated_text']
# #     return result
# #
# # # Example usage
# # resume_text = """
# # Responsibilities:
# # Build AI-Ops solutions like noise reduction, self-resolution, and chatbots using ML models (Linear SVC), BERT, and RASA.
# # Implement PDF extraction and retrieval using LLMs, RAG, and FAISS to automate SOP generation.
# # Develop end-to-end automation workflows and scripts to streamline document processing and client operations.
# # Create model evaluation tools and work with neural networks (LSTM, image datasets, MNIST).
# # Collaborate with teams to translate business requirements into technical solutions.
# # Skills & Tools:
# # Python, scikit-learn, TensorFlow, PyTorch
# # NLP (BERT), RASA, AI-Ops frameworks
# # LLMs, RAG, FAISS
# # Automation: BeautifulSoup, Selenium, Pandas
# # """
# #
# # print(clean_resume_with_llm(resume_text))
#
#
#
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
#
# model_id = "google/flan-t5-xl"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
#
# pipe = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer
# )
#
# text = """Python, scikit-learn, built, TensorFlow, developed, PyTorch, made by RASA and FAISS"""
#
# prompt = f"""
# Remove ONLY these words: ["built", "build", "developed", "develop", "made", "by"].
# Do not rephrase or summarize. Keep all other words as they are.
# Return only the cleaned text.
#
# Text: {text}
# """
#
# out = pipe(prompt, max_new_tokens=200, temperature=0.0, do_sample=False)
# print(out[0]['generated_text'])


from transformers import pipeline

# Load BART model from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large")

def clean_text_with_llm(text: str) -> str:
    """
    Use BART to clean text by summarizing and keeping only important words.
    """
    # Add an instruction so it focuses on filtering instead of making text too short
    prompt = f"Remove irrelevant or filler words but keep the key skills, experience, and entities:\n{text}"

    result = summarizer(
        prompt,
        max_length=150,   # you can tune this
        min_length=50,    # to prevent over-shortening
        do_sample=False
    )

    return result[0]['summary_text']

# Example
resume_text = """
Responsibilities:
Build AI-Ops solutions like noise reduction, self-resolution, and chatbots using ML models (Linear SVC), BERT, and RASA.
Implement PDF extraction and retrieval using LLMs, RAG, and FAISS to automate SOP generation.
Develop end-to-end automation workflows and scripts to streamline document processing and client operations.
Create model evaluation tools and work with neural networks (LSTM, image datasets, MNIST).
Collaborate with teams to translate business requirements into technical solutions.
Skills & Tools:
Python, scikit-learn, TensorFlow, PyTorch
NLP (BERT), RASA, AI-Ops frameworks
LLMs, RAG, FAISS
Automation: BeautifulSoup, Selenium, Pandas
"""

cleaned = clean_text_with_llm(resume_text)
print("Cleaned Resume:\n", cleaned)
