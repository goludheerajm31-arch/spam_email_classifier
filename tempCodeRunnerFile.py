"""
evaluate.py

Runs evaluation on the saved spam classifier model and prints
basic performance metrics on the test split.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocess import clean_text
from train import load_dataset
from utils import load_model, print_separator, print_error


def evaluate_model():
    """Load model + dataset and show evaluation metrics."""

    print()
    print_separator()
    print("  📊 Model Evaluation")
    print_separator()

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------
    print("\n  Loading saved model...")
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        print_error(str(e))
        return

    # --------------------------------------------------
    # Load dataset again (same one used for training)
    # --------------------------------------------------
    print("  Preparing dataset...")
    try:
        df = load_dataset()
    except Exception as e:
        print_error(f"Dataset load failed: {e}")
        return

    # clean text column
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    X = df["clean_text"]
    y = df["label"]

    # recreate same split as training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # vectorize test samples
    X_test_vec = vectorizer.transform(X_test)

    # predictions
    y_pred = model.predict(X_test_vec)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,
                                pos_label="spam", zero_division=0)
    recall = recall_score(y_test, y_pred,
                          pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, y_pred,
                  pos_label="spam", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])
    tn, fp, fn, tp = cm[1][1], cm[1][0], cm[0][1], cm[0][0]

    _print_metrics(
        accuracy, precision, recall, f1,
        tp, tn, fp, fn,
        y_test, y_pred
    )


# ------------------------------------------------------------------


def _metric_bar(value, width=20):
    """Simple progress bar for metric visualization."""
    filled = int(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _print_metrics(accuracy, precision, recall, f1,
                   tp, tn, fp, fn, y_test, y_pred):

    total = len(y_test)

    print()
    print_separator("═")
    print("  Classification Metrics (spam = positive)")
    print_separator("═")

    metrics = [
        ("Accuracy", accuracy, "overall correct predictions"),
        ("Precision", precision, "how reliable spam predictions are"),
        ("Recall", recall, "how much spam was detected"),
        ("F1 Score", f1, "balance between precision & recall"),
    ]

    for name, value, desc in metrics:
        bar = _metric_bar(value)
        print(f"\n  {name:<12}: {value:.4f}  {bar}  {value*100:.1f}%")
        print(f"               {desc}")

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    print()
    print_separator("─")
    print("  Confusion Matrix")
    print_separator("─")

    print("                  Predicted")
    print("                  SPAM      HAM")
    print(f"  Actual  SPAM  [ TP={tp:>3}    FN={fn:>3} ]")
    print(f"          HAM   [ FP={fp:>3}    TN={tn:>3} ]")

    print()
    print(f"  Total samples      : {total}")
    print(f"  Correct predictions: {tp + tn} ({(tp+tn)/total*100:.1f}%)")
    print(f"  Missed spam (FN)   : {fn}")
    print(f"  False alarms (FP)  : {fp}")

    # sklearn detailed report
    print()
    print_separator("─")
    print("  Detailed Classification Report")
    print_separator("─")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["spam", "ham"],
        zero_division=0,
    )

    for line in report.splitlines():
        print(f"  {line}")

    print()
    print_separator("═")
    print("  ✅ Evaluation finished")
    print_separator("═")
    print()


# run directly
if __name__ == "__main__":
    evaluate_model()