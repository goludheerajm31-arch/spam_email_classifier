"""
train.py

Responsible for training the spam detection model.
Steps:
1. Load dataset (create demo one if missing)
2. Clean text
3. Train/test split
4. TF-IDF feature extraction
5. Train Naive Bayes model
6. Save trained files
"""

import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from preprocess import clean_text
from utils import (
    DATASET_PATH,
    ensure_directories,
    save_model,
    print_separator,
)

# -------------------------------------------------------------------
# Demo dataset (used if user has no dataset)
# -------------------------------------------------------------------

DEMO_EMAILS = [
    ("Congratulations! You have won a FREE iPhone. Click here to claim now!", "spam"),
    ("URGENT: Your bank account has been compromised. Verify immediately!", "spam"),
    ("Free money! Make $5000 a week working from home.", "spam"),
    ("You are selected for a $1000 gift card. Act now!", "spam"),
    ("Hot singles in your area are waiting.", "spam"),
    ("Lose 30 pounds in 30 days with this miracle pill.", "spam"),
    ("Your PayPal account is limited. Click to restore access.", "spam"),
    ("Earn easy cash online from home.", "spam"),
    ("Dear winner, send details to claim prize.", "spam"),
    ("Cheap medication shipped fast.", "spam"),

    ("Hi John, can we schedule the meeting for Thursday?", "ham"),
    ("Please find attached the quarterly report.", "ham"),
    ("Reminder: Team standup tomorrow at 10am.", "ham"),
    ("Could you send the updated budget file?", "ham"),
    ("Looking forward to the conference next week.", "ham"),
    ("Your package has been shipped.", "ham"),
    ("Can we move our meeting to Monday?", "ham"),
    ("Client approved the proposal.", "ham"),
    ("Happy birthday! Have a great day.", "ham"),
    ("Thanks for your application. We will get back soon.", "ham"),
]


def generate_demo_dataset():
    """Creates a small dataset so project runs instantly."""
    ensure_directories()

    df = pd.DataFrame(DEMO_EMAILS, columns=["text", "label"])
    df.to_csv(DATASET_PATH, index=False)

    print(f"[INFO] Demo dataset created → {DATASET_PATH}")
    print(f"       Total samples: {len(df)}")


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------

def load_dataset():
    """Loads CSV dataset or generates demo one."""
    if not os.path.exists(DATASET_PATH):
        print("\nDataset not found. Creating demo dataset...")
        generate_demo_dataset()

    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    before = len(df)
    df = df.dropna(subset=["text", "label"])

    if len(df) != before:
        print(f"[WARN] Removed {before - len(df)} invalid rows")

    print(f"[OK] Loaded {len(df)} emails")
    return df


# -------------------------------------------------------------------
# Training pipeline
# -------------------------------------------------------------------

def train_model():

    print_separator()
    print("Starting model training...")
    print_separator()

    # 1. Load data
    print("\n[1] Loading dataset")
    df = load_dataset()

    # 2. Clean text
    print("[2] Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    print(f"    usable samples: {len(df)}")

    # 3. Split
    print("[3] Train/Test split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print(f"    train={len(X_train)}, test={len(X_test)}")

    # 4. Vectorization
    print("[4] TF-IDF vectorization")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"    features learned: {len(vectorizer.vocabulary_)}")

    # 5. Train model
    print("[5] Training Naive Bayes model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    # Save files
    print("\nSaving model...")
    save_model(model, vectorizer)

    print_separator()
    print("Training finished successfully.")
    print_separator()

    return model, vectorizer, X_test_vec, y_test


# -------------------------------------------------------------------

if __name__ == "__main__":
    train_model()