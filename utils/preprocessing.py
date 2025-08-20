import re

import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake

nltk.download('stopwords')


def extract_keywords(text):
    """
    Extract keywords from input text by applying a multi-step process:

    1. Extracts potential skills/keywords using RAKE.
    2. Removes common stopwords to retain meaningful terms.
    3. Deduplicates repeated words to ensure uniqueness.
    """
    text = extract_skills_rake(text)
    text = remove_stopwords(text)
    keywords_text = deduplicate_words(text)
    return keywords_text


def deduplicate_words(text):
    """
    Remove repeated words, keep only unique words.
    """
    seen = set()
    unique_words = [w for w in text.split() if not (w in seen or seen.add(w))]
    return " ".join(unique_words)


def extract_skills_rake(text):
    """
    Extract keywords/skills from job description text using RAKE.
    """
    r = Rake()
    r.extract_keywords_from_text(text)
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
    all_stopwords = english_stopwords | resume_stopwords

    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()
    filtered = [w for w in words if w not in all_stopwords]
    return " ".join(filtered)


def preprocess_text(text):
    """
    Preprocess raw text by normalizing formatting, case, and removing noise.

    Steps:
        1. Strip leading/trailing spaces.
        2. Lowercase text.
        3. Collapse multiple spaces/newlines.
        4. Remove non-ASCII characters.
        5. Remove punctuation.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII characters
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text
