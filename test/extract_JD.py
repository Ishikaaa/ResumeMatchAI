from rake_nltk import Rake

def extract_skills_rake(jd_text, top_k=20):
    r = Rake()
    r.extract_keywords_from_text(jd_text)
    phrases = r.get_ranked_phrases()[:top_k]
    return phrases

jd_text = """
Responsibilities:
Build AI-Ops solutions like noise reduction, self-resolution, and chatbots using ML models (Linear SVC), BERT, and RASA.
2 years of experience in Python
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

# ans = extract_skills_rake(jd_text)
# print(ans)

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(jd_text)
skills = [ent.text for ent in doc.ents if ent.label_ in ("ORG","PRODUCT","WORK_OF_ART")]
print(skills)
