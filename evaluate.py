"""
evaluate.py

Evaluates the trained spam classifier using the test split
and prints common ML metrics like accuracy, precision, recall and F1.
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


# ---------------------------------------------------------------------

def evaluate_model():
    """
    Loads saved model + dataset and checks performance on test data.
    The same random_state is used so evaluation stays consistent.
    """

    print()
    print_separator()
    print(" MODEL EVALUATION")
    print_separator()

    # ----- load trained model -----
    print("\nLoading model...")
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as err:
        print_error(str(err))
        return

    # ----- load dataset -----
    print("Preparing dataset...")
    try:
        df = load_dataset()
    except Exception as err:
        print_error(f"Dataset load failed: {err}")
        return

    # basic preprocessing (same as training)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    X = df["clean_text"]
    y = df["label"]

    # recreate train/test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # vectorize + predict
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # ----- metrics -----
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])

    # unpack values (spam treated as positive class)
    tp, fn = cm[0]
    fp, tn = cm[1]

    _print_metrics(
        accuracy, precision, recall, f1,
        tp, tn, fp, fn,
        y_test, y_pred
    )


# ---------------------------------------------------------------------

def _metric_bar(value, width=20):
    """Small ASCII bar for visual metric display."""
    filled = int(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _print_metrics(accuracy, precision, recall, f1,
                   tp, tn, fp, fn, y_test, y_pred):

    total = len(y_test)

    print()
    print_separator("=")
    print(" CLASSIFICATION METRICS")
    print_separator("=")

    metric_list = [
        ("Accuracy", accuracy, "overall correctness"),
        ("Precision", precision, "spam predictions that were correct"),
        ("Recall", recall, "spam emails successfully detected"),
        ("F1 Score", f1, "balance between precision & recall"),
    ]

    for name, value, desc in metric_list:
        bar = _metric_bar(value)
        print(f"\n{name:<10}: {value:.4f} {bar} ({value*100:.1f}%)")
        print(f"  -> {desc}")

    # ----- confusion matrix -----
    print()
    print_separator("-")
    print("CONFUSION MATRIX")
    print_separator("-")

    print("            Predicted")
    print("            SPAM   HAM")
    print(f"Actual SPAM [TP={tp:>3} FN={fn:>3}]")
    print(f"Actual HAM  [FP={fp:>3} TN={tn:>3}]")

    print(f"\nTotal samples : {total}")
    print(f"Correct       : {tp + tn} ({(tp+tn)/total*100:.1f}%)")
    print(f"False alarms  : {fp}")
    print(f"Missed spam   : {fn}")

    # ----- sklearn report -----
    print()
    print_separator("-")
    print("SKLEARN REPORT")
    print_separator("-")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["spam", "ham"],
        zero_division=0,
    )

    for line in report.splitlines():
        print(" ", line)

    print()
    print_separator("=")
    print("Evaluation finished.")
    print_separator("=")
    print()


# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_model()