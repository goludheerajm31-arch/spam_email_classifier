# Spam & Phishing Email Classifier — CLI

---

## Student Profile

| | |
|---|---|
| **Name** | Dheeraj Kumar Mahto |
| **Enrollment No.** | [Your Enrollment / Student ID] |
| **University** | [Your University / College Name] |
| **Department** | [Your Department — e.g. B.Tech CSE / BCA / MCA] |
| **Semester / Year** | [e.g. 5th Semester / 3rd Year] |
| **Course** | Fundamentals of Artificial Intelligence and Machine Learning |
| **Guide / Professor** | [Professor's Name] |

---

Spam filters have been around for decades, but building one from scratch is still one of the best ways to understand how text classification actually works under the hood. That's what this project does — a terminal-based tool that trains a Naive Bayes model on labelled email data, saves it, and lets you run predictions either by typing text or pointing it at a file.

No web interface, no notebooks. Just Python and the command line.

---

## How it works (the short version)

The model can't read text the way a person does, so the first job is converting emails into numbers. I used **TF-IDF** (Term Frequency–Inverse Document Frequency), which scores each word based on how often it shows up in a given email *and* how rare it is across the whole dataset. A word like "free" appearing repeatedly in one email but rarely in others gets a high score — and that's exactly the kind of signal that separates spam from normal mail.

Once the emails are vectorised, a **Multinomial Naive Bayes** classifier learns the statistical patterns — which words are more likely in spam, which are more common in legitimate email. It's a simple algorithm, but it's surprisingly good at this task and trains in under a second even on larger datasets.

The full flow looks like this:

```
emails.csv  →  clean text  →  TF-IDF matrix  →  Naive Bayes  →  model.pkl
                                                                      ↓
                                              new email  →  SPAM or HAM + confidence %
```

The trained model gets saved to disk, so you only need to train once. After that, predictions are instant.

---

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
