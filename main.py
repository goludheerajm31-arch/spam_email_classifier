"""
main.py

Entry file for Spam & Phishing Email Classifier CLI
Run: python main.py

Simple terminal menu:
1 - Train model
2 - Evaluate model
3 - Classify typed email
4 - Classify email from file
5 - Help
6 - Exit
"""

import sys

from utils import (
    ensure_directories,
    print_banner,
    print_separator,
    print_invalid_command,
    print_error,
    get_non_empty_input,
)

from train import train_model
from evaluate import evaluate_model
from predict import predict_text, predict_from_file


# ---------------------- MENU UI ----------------------

def display_menu():
    """Prints main menu options."""
    print_separator("─")
    print(" MAIN MENU")
    print_separator("─")
    print(" 1. Train Model")
    print(" 2. Evaluate Model")
    print(" 3. Classify Email (paste text)")
    print(" 4. Classify from File")
    print(" 5. Help")
    print(" 6. Exit")
    print_separator("─")


# ---------------------- ACTION HANDLERS ----------------------

def handle_train():
    """Start training pipeline."""
    try:
        train_model()
    except Exception as err:
        print_error(f"Training error: {err}")


def handle_evaluate():
    """Run evaluation on saved model."""
    try:
        evaluate_model()
    except Exception as err:
        print_error(f"Evaluation error: {err}")


def handle_classify_text():
    """Take multiline email input from user."""
    print()
    print_separator("─")
    print(" CLASSIFY EMAIL")
    print_separator("─")
    print("Paste your email below.")
    print("Press Enter twice when done.\n")

    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break

        # stop when user presses blank line after content
        if line == "":
            if lines:
                break
            continue

        lines.append(line)

    email_text = "\n".join(lines).strip()

    if not email_text:
        print_error("No email text entered.")
        return

    predict_text(email_text)


def handle_classify_file():
    """Classify email from a text file."""
    print()
    print_separator("─")
    print(" CLASSIFY FROM FILE")
    print_separator("─")

    path = get_non_empty_input("File path: ")
    predict_from_file(path)


def handle_help():
    """Show usage instructions."""
    print()
    print_separator("=")
    print(" HELP")
    print_separator("=")

    print("""
This project detects spam/phishing emails using ML.

Workflow:
1. Train the model first
2. Classify emails
3. Evaluate performance

Concepts used:
- TF-IDF vectorization
- Naive Bayes classifier
- Train/Test split

Quick start:
python main.py -> option 1 (train)
python main.py -> option 3 (classify)
python main.py -> option 2 (evaluate)
""")

    print_separator("=")
    print()


def handle_exit():
    """Exit program safely."""
    print("\nThanks for using the Spam Classifier. Bye!\n")
    sys.exit(0)


# ---------------------- MAIN LOOP ----------------------

def main():
    """Program entry point."""
    ensure_directories()
    print_banner()

    actions = {
        "1": handle_train,
        "2": handle_evaluate,
        "3": handle_classify_text,
        "4": handle_classify_file,
        "5": handle_help,
        "6": handle_exit,
    }

    while True:
        try:
            display_menu()
            choice = input("Enter choice (1-6): ").strip()

            if choice in actions:
                actions[choice]()
            else:
                print_invalid_command()

        except KeyboardInterrupt:
            print()
            handle_exit()

        except Exception as err:
            print_error(f"Unexpected error: {err}")
            print("Returning to menu...\n")


# ---------------------- RUN SCRIPT ----------------------

if __name__ == "__main__":
    main()