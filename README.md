# 🛡️ Spam Email Classifier — CLI Application

**Course:** Fundamentals of Artificial Intelligence and Machine Learning  
**Project Type:** End-to-End Machine Learning CLI Application  
**Language:** Python 3.8+

---

## 📌 Project Description

This command-line application uses **Machine Learning** to automatically detect whether an email is **Spam / Phishing** or **Legitimate (Ham)**. It demonstrates a complete AI/ML workflow — from raw text data to a trained model that makes real predictions — all controlled through an interactive terminal menu.

---

## 🧠 AI / ML Concepts Explained

| Concept | Explanation |
|---|---|
| **TF-IDF** | Converts words into numbers. Words rare across emails but frequent in one email score higher, making them stronger features. |
| **Multinomial Naive Bayes** | A probabilistic classifier ideal for text. It learns which words appear more in spam vs. ham and uses those patterns for predictions. |
| **Train / Test Split** | 80% of emails train the model; 20% are held back to test it fairly on unseen data. |
| **Stopword Removal** | Common words (the, is, at, a) are removed because they carry no spam/ham signal. |
| **Text Preprocessing** | Lowercasing, punctuation removal, and number removal ensure consistent input to the model. |
| **Pickle Serialisation** | The trained model is saved to disk so it doesn't need to be retrained on every run. |

---

## 📁 Folder Structure

```
spam-email-classifier-cli/
│
├── data/
│   └── emails.csv          ← Training dataset (text, label)
│
├── models/
│   ├── model.pkl           ← Saved trained classifier (auto-generated)
│   └── vectorizer.pkl      ← Saved TF-IDF vectorizer (auto-generated)
│
├── main.py                 ← CLI entry point and menu loop
├── train.py                ← Training pipeline
├── predict.py              ← Prediction functions
├── preprocess.py           ← Text cleaning utilities
├── evaluate.py             ← Model evaluation and metrics
├── utils.py                ← Shared helpers and constants
│
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## ⚙️ Installation

### 1. Clone / download the project
```bash
# If using git:
git clone <repository-url>
cd spam-classifier-cli

# Or simply extract the zip and navigate to the folder:
cd spam-classifier-cli
```

### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** NLTK stopwords are downloaded automatically the first time you run the application. No manual setup required.

---

## 🚀 How to Run

```bash
python main.py
```

You will see:

```
════════════════════════════════════════════════════
  🛡️  Spam & Phishing Email Classifier CLI
      Fundamentals of AI and Machine Learning
════════════════════════════════════════════════════

────────────────────────────────────────────────────
  MAIN MENU
────────────────────────────────────────────────────
  1.  Train Model
  2.  Evaluate Model
  3.  Classify an Email  (type or paste text)
  4.  Classify from File (.txt file path)
  5.  Help
  6.  Exit
────────────────────────────────────────────────────
  Enter choice (1-6):
```

---

## 📋 Commands Reference

| Option | Action | Description |
|---|---|---|
| **1** | Train Model | Loads dataset, preprocesses text, trains Naive Bayes, saves model files |
| **2** | Evaluate Model | Tests the saved model; shows Accuracy, Precision, Recall, F1, Confusion Matrix |
| **3** | Classify an Email | Type / paste email text; get instant SPAM or HAM verdict with confidence % |
| **4** | Classify from File | Provide path to a `.txt` file; the content is classified automatically |
| **5** | Help | Detailed usage guide and ML concept explanations |
| **6** | Exit | Safely exits the application |

---

## 🔄 ML Workflow (Step by Step)

```
  Raw Email Data (emails.csv)
        │
        ▼
  Text Preprocessing  ←──── preprocess.py
  (lowercase, remove punctuation, stopwords)
        │
        ▼
  TF-IDF Vectorisation  ←── train.py / scikit-learn
  (text → numerical feature matrix)
        │
        ▼
  Model Training  ←────────── train.py
  (Multinomial Naive Bayes, random_state=42)
        │
        ▼
  Save Model & Vectorizer  ←── utils.py / pickle
  (models/model.pkl, models/vectorizer.pkl)
        │
        ▼
  New Email Input  ←────────── main.py / predict.py
        │
        ▼
  Same Preprocessing + Transform
        │
        ▼
  Prediction → SPAM or HAM  + Confidence %
```

---

## 💻 Example Usage Sessions

### Training the model
```
Enter choice (1-6): 1

  ────────────────────────────────────────────────────
    🚀  MODEL TRAINING PIPELINE
  ────────────────────────────────────────────────────

  [1/5] Loading dataset...
  [✓] Dataset loaded: 50 emails ({'ham': 25, 'spam': 25})

  [2/5] Cleaning and preprocessing text...
        50 valid samples after cleaning.

  [3/5] Splitting data (80% train / 20% test)...
        Train samples: 40 | Test samples: 10

  [4/5] Fitting TF-IDF vectorizer...
        Vocabulary size: 312 features

  [5/5] Training Multinomial Naive Bayes classifier...

  Saving model files...
  [✓] Model saved     → models/model.pkl
  [✓] Vectorizer saved → models/vectorizer.pkl

  ✅  Training complete! Model is ready to use.
```

### Classifying an email
```
Enter choice (1-6): 3

  ✉️   CLASSIFY AN EMAIL
  ────────────────────────────────────────────────────
  Email text (blank line to finish):
  > Congratulations! You have won a FREE iPhone. Click here now!
  >

  ────────────────────────────────────────────────────
  🚨  CLASSIFICATION RESULT
  ────────────────────────────────────────────────────
  Source     : Direct Input
  Verdict    : SPAM
  Confidence : 98.7%  [████████████████████]
  ────────────────────────────────────────────────────
  ⚠️  This email shows signs of spam or phishing.
      Do NOT click links or share personal information.
```

### Evaluating the model
```
Enter choice (1-6): 2

  CLASSIFICATION METRICS  (positive class = spam)
  ════════════════════════════════════════════════

  Accuracy     : 0.9000  [██████████████████░░]  90.0%
  Precision    : 1.0000  [████████████████████]  100.0%
  Recall       : 0.8000  [████████████████░░░░]  80.0%
  F1 Score     : 0.8889  [█████████████████░░░]  88.9%
```

---

## 📂 Dataset Format

The CSV file (`data/emails.csv`) must have exactly two columns:

```csv
text,label
"Congratulations! You won a free iPhone!",spam
"Meeting scheduled for Thursday at 3pm",ham
```

- `text`  — The raw email body  
- `label` — Either `spam` or `ham`

> A 50-email demo dataset is **included** and also **auto-generated** if the file is missing.

---

## 🛡️ Error Handling

The application gracefully handles:

| Scenario | Response |
|---|---|
| Dataset missing | Auto-generates a 50-email demo dataset |
| Model not trained | Friendly message: "Please run Train Model first" |
| Empty email input | Prompts user to re-enter |
| Invalid menu choice | Displays "Invalid command! Please try again." |
| File not found | Clear error with the attempted path |
| `Ctrl+C` | Graceful exit with goodbye message |

---

## 🔧 Customising the Dataset

To use your own data:
1. Prepare a CSV file with `text` and `label` columns
2. Replace `data/emails.csv` with your file
3. Run **Option 1 (Train Model)** to retrain on your data

For best results, use at least **200+ labelled emails** with balanced spam/ham distribution.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `scikit-learn` | TF-IDF vectoriser, Naive Bayes classifier, evaluation metrics |
| `pandas` | Loading and manipulating the CSV dataset |
| `numpy` | Numerical operations (used internally by scikit-learn) |
| `nltk` | English stopword list for text preprocessing |

Install all at once: `pip install -r requirements.txt`

---

## 👨‍💻 Module Responsibilities

| File | Responsibility |
|---|---|
| `main.py` | CLI menu loop, command routing, keyboard interrupt handling |
| `train.py` | Dataset loading, preprocessing, TF-IDF training, model saving |
| `predict.py` | Loading saved model, classifying text/file input, displaying results |
| `preprocess.py` | `clean_text()` — the single reusable text cleaning function |
| `evaluate.py` | Accuracy, Precision, Recall, F1 Score, Confusion Matrix display |
| `utils.py` | Path constants, directory creation, model I/O, input validation, error messages |
