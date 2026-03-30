# 🛡️ Spam Email Classifier — CLI Application

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Project Type](https://img.shields.io/badge/Project-CLI%20Application-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 👨‍🎓 Student Profile

**Name:** Dheeraj  
**Program:** B.Tech Engineering  
**Course:** Fundamentals of Artificial Intelligence and Machine Learning  
**Project Type:** Academic Machine Learning Project  

This project was built to understand how Machine Learning can be applied to real-world cybersecurity problems such as spam and phishing email detection.

---

## 📌 Project Overview

Spam and phishing emails are a major cybersecurity threat. This project presents a **Command-Line Interface (CLI) application** that automatically detects whether an email is:

✅ **HAM (Legitimate Email)**  
🚨 **SPAM (Malicious / Phishing Email)**  

The application demonstrates a complete **end-to-end Machine Learning pipeline**, including:

- Data preprocessing
- Feature extraction
- Model training
- Performance evaluation
- Real-time prediction

All operations are controlled through an interactive terminal menu.

---

## ✨ Key Features

- Interactive CLI menu system
- Machine Learning based email classification
- TF-IDF text feature extraction
- Multinomial Naive Bayes model
- Model saving using Pickle
- Evaluation metrics (Accuracy, Precision, Recall, F1)
- File-based and manual email classification
- Friendly error handling

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | ML algorithms & evaluation |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| NLTK | Stopword removal |
| Pickle | Model serialization |

---

## 📁 Project Structure


## Project structure

```
spam-classifier-cli/
│
├── data/
│   └── emails.csv          training dataset
│
├── models/
│   ├── model.pkl           saved classifier  (created after training)
│   └── vectorizer.pkl      saved TF-IDF vectorizer  (created after training)
│
├── main.py                 the only file you need to run
├── train.py                training pipeline
├── predict.py              prediction logic
├── preprocess.py           text cleaning
├── evaluate.py             metrics and confusion matrix
├── utils.py                shared helpers, path constants, error messages
│
├── requirements.txt
└── README.md
```

The `models/` and `data/` folders are created automatically if they don't exist. If `emails.csv` is missing, the app generates a 50-email demo dataset so you can test it right away.

---

## Getting started

```bash
# 1. Clone the repo
git clone <repository-url>
cd spam-classifier-cli

# 2. Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python main.py
```

NLTK stopwords download automatically on the first run — you don't need to do anything extra.

---

## Running the app

```bash
python main.py
```

```
════════════════════════════════════════════════════
  🛡️  Spam & Phishing Email Classifier CLI
      Fundamentals of AI and Machine Learning
════════════════════════════════════════════════════

  MAIN MENU
  ──────────────────────────────────────────────────
  1.  Train Model
  2.  Evaluate Model
  3.  Classify an Email  (type or paste text)
  4.  Classify from File (.txt file path)
  5.  Help
  6.  Exit
  ──────────────────────────────────────────────────
  Enter choice (1-6):
```

**Start with option 1.** Training takes a couple of seconds and saves the model files. After that, options 2–4 are available.

---

## What each option does

**1 — Train Model**
Loads `data/emails.csv`, cleans the text, fits the TF-IDF vectorizer on 80% of the data, trains Naive Bayes on the result, and saves both `model.pkl` and `vectorizer.pkl` to the `models/` folder.

**2 — Evaluate Model**
Runs the saved model against the held-out 20% test split and prints accuracy, precision, recall, and F1 score — plus a confusion matrix breaking down true/false positives and negatives.

**3 — Classify an Email**
Prompts you to paste or type an email. Hit Enter on a blank line when done. You get a verdict (SPAM or HAM) and a confidence percentage.

**4 — Classify from File**
Give it a path to any `.txt` file and it reads and classifies the content automatically.

**5 — Help**
In-terminal guide covering ML concepts, workflow, and usage tips.

---

## Sample output

Training:
```
  [1/5] Loading dataset...
  [✓] Dataset loaded: 50 emails ({'spam': 25, 'ham': 25})

  [2/5] Cleaning and preprocessing text...
        50 valid samples after cleaning.

  [3/5] Splitting data (80% train / 20% test)...
        Train samples: 40 | Test samples: 10

  [4/5] Fitting TF-IDF vectorizer...
        Vocabulary size: 400 features

  [5/5] Training Multinomial Naive Bayes classifier...

  [✓] Model saved     → models/model.pkl
  [✓] Vectorizer saved → models/vectorizer.pkl
  ✅  Training complete!
```

Classifying:
```
  Email text (blank line to finish):
  > Congratulations! You have won a FREE iPhone. Click here now!
  >

  🚨  CLASSIFICATION RESULT
  ──────────────────────────────────────────────────
  Verdict    : SPAM
  Confidence : 98.7%  [████████████████████]
  ──────────────────────────────────────────────────
  ⚠️  Do NOT click links or share personal information.
```

Evaluation on the 50-email demo dataset:
```
  Accuracy  : 0.9000  [██████████████████░░]  90.0%
  Precision : 0.8333  [████████████████░░░░]  83.3%
  Recall    : 1.0000  [████████████████████]  100.0%
  F1 Score  : 0.9091  [██████████████████░░]  90.9%
```

A 100% recall on spam means the model caught every single spam email in the test set — zero missed. The one false alarm (a legitimate email flagged as spam) is what pulls precision below perfect. For a 50-email training set, that's a reasonable tradeoff.

---

## Using your own dataset

The app expects a CSV with two columns: `text` (the email body) and `label` (either `spam` or `ham`). Replace `data/emails.csv` with your file and retrain with option 1.

```csv
text,label
"Win a free vacation now — limited spots!",spam
"The project review is set for Tuesday at 2pm.",ham
```

More data improves results meaningfully — the included dataset is intentionally small for portability. A few hundred balanced examples is a reasonable starting point for real testing.

---

## Dependencies

- `scikit-learn` — TF-IDF vectorizer, Naive Bayes, evaluation metrics
- `pandas` — loading the CSV
- `numpy` — underlying numerical operations
- `nltk` — English stopword list (auto-downloaded; the app has a built-in fallback if unavailable)

---

## Notes on a few design choices

I kept preprocessing simple on purpose — lowercase, remove punctuation and numbers, strip stopwords. More aggressive approaches like stemming or lemmatisation don't always help with Naive Bayes and add complexity for marginal gains on this kind of task.

`random_state=42` in both the train/test split and the classifier means results are fully reproducible across runs. The split also uses `stratify=y` so spam and ham stay proportionally balanced between train and test.

Bigrams (`ngram_range=(1, 2)`) are included in the vectorizer. The phrase "guaranteed income" is a stronger spam signal than either word alone — bigrams help capture patterns like that.
