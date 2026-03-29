"""
preprocess.py

Basic text cleaning used before training and prediction.
Converts raw email text into a simpler format for the model.
"""

import re
import string


# ---------------------------------------------------------------------
# Stopwords loader
# Try using NLTK if available, otherwise fall back to a built-in list.
# This avoids breaking the project if NLTK data is missing.
# ---------------------------------------------------------------------

def _load_stopwords():
    try:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            print("[INFO] Downloading stopwords (first run only)...")
            nltk.download("stopwords", quiet=True)

        from nltk.corpus import stopwords
        return set(stopwords.words("english"))

    except Exception:
        # fallback list (common English words)
        return {
            "i","me","my","myself","we","our","ours","ourselves",
            "you","your","yours","yourself","yourselves","he","him",
            "his","himself","she","her","hers","herself","it","its",
            "itself","they","them","their","theirs","themselves",
            "what","which","who","whom","this","that","these","those",
            "am","is","are","was","were","be","been","being","have",
            "has","had","having","do","does","did","doing","a","an",
            "the","and","but","if","or","because","as","until","while",
            "of","at","by","for","with","about","against","between",
            "into","through","during","before","after","above","below",
            "to","from","up","down","in","out","on","off","over","under",
            "again","further","then","once","here","there","when",
            "where","why","how","all","both","each","few","more","most",
            "other","some","such","no","nor","not","only","own","same",
            "so","than","too","very","can","will","just","should","now"
        }


# load once when module imports
STOP_WORDS = _load_stopwords()


# ---------------------------------------------------------------------

def clean_text(text):
    """
    Simple preprocessing pipeline used by both training and prediction.
    """

    if not isinstance(text, str):
        return ""

    # lowercase everything
    text = text.lower()

    # remove links
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    return " ".join(words)