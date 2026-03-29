"""
predict.py

Handles email classification using the saved spam model.
You can classify text directly or load it from a file.
"""

import os

from preprocess import clean_text
from utils import load_model, print_separator, print_error


# ---------------------------------------------------------------------

def _load_and_predict(text):
    """
    Internal helper used by both prediction modes.
    Loads model, cleans text and returns prediction + confidence.
    """

    # load trained components
    model, vectorizer = load_model()

    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        raise ValueError("Email content became empty after preprocessing.")

    # convert text → features
    features = vectorizer.transform([cleaned_text])

    # prediction
    label = model.predict(features)[0]

    # probability for confidence score
    probs = model.predict_proba(features)[0]
    class_list = list(model.classes_)
    confidence = probs[class_list.index(label)] * 100

    return label, confidence


# ---------------------------------------------------------------------

def predict_text(text):
    """Classify email text entered directly by the user."""

    if not text or not text.strip():
        print_error("Empty email text.")
        return None

    try:
        label, confidence = _load_and_predict(text)
        _show_result(label, confidence, "Direct Input")
        return label

    except FileNotFoundError as err:
        print_error(str(err))
    except ValueError as err:
        print_error(str(err))
    except Exception as err:
        print_error(f"Prediction error: {err}")

    return None


# ---------------------------------------------------------------------

def predict_from_file(filepath):
    """Reads a text file and classifies its contents."""

    if not filepath or not filepath.strip():
        print_error("Please provide a file path.")
        return None

    filepath = filepath.strip()

    if not os.path.isfile(filepath):
        print_error(f"File not found: {filepath}")
        return None

    # ----- read file -----
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # fallback encoding (common in downloaded files)
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as err:
            print_error(f"Unable to read file: {err}")
            return None
    except Exception as err:
        print_error(f"Error opening file: {err}")
        return None

    if not content.strip():
        print_error("File is empty.")
        return None

    # quick preview
    preview = content[:120].replace("\n", " ")
    suffix = "..." if len(content) > 120 else ""

    print(f"\nFile: {filepath}")
    print(f"Preview: \"{preview}{suffix}\"")

    # ----- prediction -----
    try:
        label, confidence = _load_and_predict(content)
        _show_result(label, confidence, os.path.basename(filepath))
        return label

    except FileNotFoundError as err:
        print_error(str(err))
    except ValueError as err:
        print_error(str(err))
    except Exception as err:
        print_error(f"Prediction error: {err}")

    return None


# ---------------------------------------------------------------------

def _show_result(label, confidence, source=""):
    """Displays prediction result nicely in terminal."""

    spam_detected = label.lower() == "spam"

    icon = "🚨" if spam_detected else "✅"
    verdict = "SPAM" if spam_detected else "HAM (Legitimate)"

    bar_length = int(confidence / 5)  # 20-char bar
    bar_char = "█" if spam_detected else "░"

    print()
    print_separator("-")
    print(f"{icon} CLASSIFICATION RESULT")
    print_separator("-")

    if source:
        print(f"Source     : {source}")

    print(f"Verdict    : {verdict}")
    print(f"Confidence : {confidence:.1f}% [{bar_char * bar_length:<20}]")

    print_separator("-")

    if spam_detected:
        print("Warning: This email looks suspicious. Avoid clicking links.")
    else:
        print("Looks safe based on model prediction.")

    print()