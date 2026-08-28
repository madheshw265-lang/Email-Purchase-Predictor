# 📧 Email Purchase Predictor

<p align="center">
  <img src="assets/Banner.png" alt="Email Purchase Predictor" width="100%">
</p>

## 🚀 Overview

**Email Purchase Predictor** is an AI-powered machine learning application that analyzes customer emails and predicts their purchase intent.

The system classifies emails into two categories:

- 🟢 **Likely to Buy**
- 🔴 **Unlikely to Buy**

It uses **Natural Language Processing (NLP)**, **TF-IDF feature extraction**, and a **Random Forest classifier**.

## ✨ Features

- 📧 Customer email analysis
- 🧠 Natural Language Processing
- 🔤 TF-IDF feature extraction
- 🌲 Random Forest classification
- 📊 Purchase intent prediction
- 📈 Confidence score
- 📋 Model evaluation
- 🖥️ Interactive Streamlit web application

## 🛠️ Tech Stack

- 🐍 Python
- 🐼 Pandas
- 🤖 Scikit-learn
- 📝 NLTK
- 🔤 TF-IDF
- 🌲 Random Forest
- 🎈 Streamlit
- 💾 Pickle

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | **80%** |
| Precision | **80%** |
| Recall | **80%** |
| F1 Score | **80%** |

## 📂 Project Structure

```text
Email-Purchase-Predictor/
│
├── assets/
│   └── Banner.png
│
├── data/
│   └── emails.csv
│
├── model/
│   ├── email_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── email_predictor.py
├── train_model.py
├── evaluate_model.py
├── test_predictor.py
├── requirements.txt
├── .gitignore
└── README.md
