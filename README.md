#  Spam Email Classifier — CLI Application


##  Student Profile

**Name:** Dheeraj Kumar Mahto
**Registration no.:** 25BCE10465 
**Course:** Fundamentals of Artificial Intelligence and Machine Learning  
**Project Repository:** https://github.com/goludheerajm31-arch/spam_email_classifier.git

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


## Dependencies

- `scikit-learn` — TF-IDF vectorizer, Naive Bayes, evaluation metrics
- `pandas` — loading the CSV
- `numpy` — underlying numerical operations
- `nltk` — English stopword list (auto-downloaded; the app has a built-in fallback if unavailable)

---

