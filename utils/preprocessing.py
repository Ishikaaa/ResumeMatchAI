from rake_nltk import Rake
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


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


def extract_skills_rake(jd_text, top_k=20):
    """
    Extract keywords/skills from job description text using RAKE.

    Args:
        jd_text (str): Job description text
        top_k (int): Number of top keywords to extract (optional)

    Returns:
        str: Keywords joined as a single string
    """
    r = Rake()
    r.extract_keywords_from_text(jd_text)
    phrases = [re.sub(r"[.)]", "", p) for p in r.get_ranked_phrases()]
    return " ".join(phrases)


def remove_stopwords(text):
    """
    Remove all English + resume-specific stopwords from text.
    """
    # General English stopwords
    english_stopwords = set(stopwords.words('english'))

    # Resume-specific stopwords
    resume_stopwords = set([
        "implemented", "developed", "working", "worked", "responsible", "handled",
        "performed", "participated", "involved", "experience", "project", "role",
        "tasks", "duties", "team", "solution", "solutions", "workflow", "process",
        "professional", "skills", "knowledge", "accomplished", "achievement", "build", "responsibilities",
        "implementations"
    ])

    # Combine both sets
    ALL_STOPWORDS = english_stopwords.union(resume_stopwords)

    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()
    filtered = [w for w in words if w not in ALL_STOPWORDS]
    return " ".join(filtered)


def preprocess_text(text):
    text = text.strip()  # remove leading/trailing spaces
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text


# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     return text.strip()
