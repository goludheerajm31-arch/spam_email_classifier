"""
utils.py

Common helper functions shared across the project.
Handles paths, saving/loading models, input checks,
and small console helpers.
"""

import os
import pickle


# ------------------------------------------------------------------
# Paths used across the project
# ------------------------------------------------------------------

DATA_DIR = "data"
MODELS_DIR = "models"

DATASET_PATH = os.path.join(DATA_DIR, "emails.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")


# ------------------------------------------------------------------
# Folder setup
# ------------------------------------------------------------------

def ensure_directories():
    """Create required folders if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# File checks
# ------------------------------------------------------------------

def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def check_dataset_exists() -> bool:
    return file_exists(DATASET_PATH)


def check_model_exists() -> bool:
    # both files must exist
    return file_exists(MODEL_PATH) and file_exists(VECTORIZER_PATH)


# ------------------------------------------------------------------
# Model save / load
# ------------------------------------------------------------------

def save_model(model, vectorizer):
    """Save trained model and vectorizer using pickle."""
    ensure_directories()

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model saved:", MODEL_PATH)
    print("Vectorizer saved:", VECTORIZER_PATH)


def load_model():
    """Load model + vectorizer from disk."""
    if not check_model_exists():
        raise FileNotFoundError(
            "Model not found. Train the model first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# ------------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------------

def get_valid_menu_choice(prompt: str, valid_choices: list) -> str:
    """Keep asking until user enters a valid option."""
    while True:
        choice = input(prompt).strip()

        if choice in valid_choices:
            return choice

        print_invalid_command()


def get_non_empty_input(prompt: str) -> str:
    """Reject empty input."""
    while True:
        text = input(prompt).strip()

        if text:
            return text

        print_error("Input cannot be empty.")


# ------------------------------------------------------------------
# Console messages
# ------------------------------------------------------------------

def print_invalid_command():
    print("\n[!] Invalid command. Try again.\n")


def print_error(message: str):
    print(f"\n[ERROR] {message}\n")


def print_success(message: str):
    print(f"\n[OK] {message}\n")


def print_separator(char: str = "-", width: int = 50):
    print(char * width)


def print_banner():
    print()
    print_separator("=")
    print(" Spam & Phishing Email Classifier (CLI)")
    print_separator("=")
    print()