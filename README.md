🛡️ Spam Email Classifier — CLI Application
Course: Fundamentals of Artificial Intelligence and Machine Learning
Project Type: End-to-End Machine Learning CLI Application
Language: Python 3.8+
👨‍🎓 Student Profile
Student Name: Dheeraj
Program: B.Tech Engineering
Course: Fundamentals of Artificial Intelligence and Machine Learning
Project Type: Academic Mini Project (CLI-based ML Application)
This project was developed as part of coursework to understand how machine learning models are applied in real-world cybersecurity problems such as spam and phishing detection.
📌 Project Overview
Email spam and phishing attacks are among the most common cybersecurity threats today. This project implements a command-line application that automatically classifies emails as Spam (malicious) or Ham (legitimate) using Machine Learning.
The goal of this project is not only to build a working classifier but also to demonstrate the complete ML workflow — from data preprocessing and training to evaluation and real-time prediction — all through a simple interactive terminal interface.
🧠 Machine Learning Concepts Used
Concept	What it Means (Simple Explanation)
TF-IDF Vectorization	Converts text into numbers so the ML model can understand email content. Important words receive higher weight.
Multinomial Naive Bayes	A lightweight algorithm well suited for text classification problems like spam detection.
Train/Test Split	Dataset is divided into training data (80%) and testing data (20%) to evaluate performance fairly.
Stopword Removal	Removes common words such as the, is, at which do not help classification.
Text Preprocessing	Cleans text by converting to lowercase and removing punctuation/noise.
Pickle Serialization	Saves trained models so they can be reused without retraining every time.
📁 Project Structure
spam-email-classifier-cli/
│
├── data/
│   └── emails.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── main.py
├── train.py
├── predict.py
├── preprocess.py
├── evaluate.py
├── utils.py
│
├── requirements.txt
└── README.md
⚙️ Installation Guide
1️⃣ Clone or Download Project
git clone <repository-url>
cd spam-classifier-cli
Or download the ZIP and open the folder in terminal.
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
Activate:
Windows
venv\Scripts\activate
macOS / Linux
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
NLTK stopwords will download automatically during first execution.
🚀 Running the Application
python main.py
You will see an interactive menu:
══════════════════════════════════════════════
  🛡️ Spam & Phishing Email Classifier CLI
══════════════════════════════════════════════

1. Train Model
2. Evaluate Model
3. Classify an Email
4. Classify from File
5. Help
6. Exit
📋 Menu Options
Option	Function
1	Train model using dataset
2	Evaluate model performance
3	Classify pasted email text
4	Classify email from .txt file
5	Help & explanations
6	Exit application
🔄 Machine Learning Workflow
Dataset (emails.csv)
        ↓
Text Cleaning & Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Naive Bayes Training
        ↓
Model Saved (pickle)
        ↓
New Email Input
        ↓
Prediction → SPAM / HAM
💻 Example Output
Training
[✓] Dataset loaded
[✓] Text preprocessing completed
[✓] Model trained successfully
[✓] Model saved to models/model.pkl
Prediction
Verdict    : SPAM
Confidence : 98.7%
⚠️ Avoid clicking suspicious links.
📂 Dataset Format
data/emails.csv
text,label
"Congratulations! You won a free iPhone!",spam
"Meeting scheduled tomorrow",ham
text → Email content
label → spam / ham
🛡️ Error Handling Features
The program safely handles:
Missing dataset (auto demo dataset created)
Model not trained warning
Invalid menu input
Empty email input
Missing files
Keyboard interrupt (Ctrl+C exit)
🔧 Custom Dataset Usage
Replace data/emails.csv
Keep columns: text,label
Run Train Model again
Recommended: minimum 200+ labelled emails for better accuracy.
📦 Libraries Used
Library	Purpose
scikit-learn	ML algorithms & evaluation
pandas	Dataset handling
numpy	Numerical processing
nltk	Stopword removal
Install using:
pip install -r requirements.txt
👨‍💻 Module Responsibilities
File	Role
main.py	CLI interface and menu logic
train.py	Training pipeline
predict.py	Email classification
preprocess.py	Text cleaning functions
evaluate.py	Performance metrics
utils.py	Shared utilities and helpers
🎯 Learning Outcomes
Through this project, I learned:
How text data is converted into machine-readable features
Training and evaluating ML models
Building modular Python applications
Creating interactive CLI tools
Applying AI concepts to cybersecurity problems
📄 License
This project is created for educational purposes as part of academic coursework.
