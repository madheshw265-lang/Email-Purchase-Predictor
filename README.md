📧 Email Purchase Predictor
<p align="center"> <img src="assets/Banner.png" alt="Email Purchase Predictor" width="100%"> </p> <p align="center"> <strong>AI-Powered Customer Purchase Intent Classification</strong> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python"> <img src="https://img.shields.io/badge/NLP-TF--IDF-orange"> <img src="https://img.shields.io/badge/Model-Random%20Forest-green"> <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit"> <img src="https://img.shields.io/badge/Accuracy-80%25-success"> </p>
📌 About the Project

Email Purchase Predictor is a Machine Learning project that predicts whether a customer is likely to purchase a product based on the content of their email.

The project uses Natural Language Processing (NLP) to process email text, TF-IDF Vectorization to convert text into numerical features, and a Random Forest Classifier to predict purchase intent.

A Streamlit web application provides an easy-to-use interface where users can enter an email and instantly receive a prediction and confidence score.

Prediction Categories

🟢 Likely to Buy

🔴 Unlikely to Buy

✨ Features
📧 Analyze customer email text
🧹 Text preprocessing
🔤 TF-IDF feature extraction
🌲 Random Forest classification
🎯 Purchase intent prediction
📊 Prediction confidence score
📈 Model evaluation
🔢 Confusion matrix
📋 Classification report
🧪 Prediction testing
🖥️ Streamlit web interface
💾 Saved trained model
📁 CSV dataset
🛠️ Technologies
Technology	Usage
Python	Programming
Pandas	Dataset processing
NumPy	Numerical operations
Scikit-learn	Machine Learning
TF-IDF	Text feature extraction
Random Forest	Classification
Joblib	Model saving/loading
Streamlit	Web application
🧠 How It Works

The project follows this pipeline:

Customer Email
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Numerical Features
      ↓
Random Forest Classifier
      ↓
Purchase Intent
      ↓
Confidence Score
1. Email Input

The user enters a customer email into the Streamlit application.

2. Text Processing

The email is cleaned and prepared for machine learning.

3. TF-IDF

TF-IDF converts the email into numerical features.

4. Random Forest

The trained Random Forest model analyzes the features.

5. Prediction

The system predicts:

Likely to Buy

or

Unlikely to Buy
6. Confidence

The application also displays the prediction confidence.

📊 Dataset

The dataset is located at:

data/emails.csv

The current dataset contains:

Total Emails      : 80
Training Samples  : 64
Testing Samples   : 16
Example Data
Email	Label
I am interested in buying your product.	Likely to Buy
Can you send me the price?	Likely to Buy
Please send the payment details.	Likely to Buy
I am not interested in this product.	Unlikely to Buy
Please stop sending me emails.	Unlikely to Buy
🏋️ Model Training

Train the model using:

python train_model.py

The training script creates:

model/
├── email_model.pkl
└── tfidf_vectorizer.pkl
Training Output
Dataset loaded successfully!
Total emails: 80

====================================
       MODEL TRAINING COMPLETE
====================================

Training samples : 64
Testing samples  : 16
Model             : Random Forest
Features          : TF-IDF

Saved files:
✓ model/email_model.pkl
✓ model/tfidf_vectorizer.pkl
📈 Model Evaluation

Evaluate the model using:

python evaluate_model.py
Performance
Metric	Score
Accuracy	80%
Precision	80%
Recall	80%
F1 Score	80%
Confusion Matrix
[[4 1]
 [1 4]]
Classification Report
                 precision    recall  f1-score   support

Unlikely to Buy       0.80      0.80      0.80         5
Likely to Buy         0.80      0.80      0.80         5

accuracy                           0.80        10
macro avg             0.80      0.80      0.80        10
weighted avg          0.80      0.80      0.80        10

Note: The current dataset is small, so the reported performance is based on a limited number of test samples.

🧪 Test the Predictor

Run:

python test_predictor.py
Example
Email:
I am interested in buying your product.
Please send me the price.

Prediction:
Likely to Buy

Another example:

Email:
I am not interested in this product.

Prediction:
Unlikely to Buy
🌐 Streamlit Web Application

Run the application using:

python -m streamlit run app.py

The application will open in your browser at:

http://localhost:8501
The application provides:
📧 Email input
🔮 Purchase prediction
📊 Confidence percentage
🤖 Model information
📈 Model performance
📸 Screenshots
🟢 Likely to Buy
<p align="center"> <img src="screenshots/likely-to-buy.png" alt="Likely to Buy" width="850"> </p>
🔴 Unlikely to Buy
<p align="center"> <img src="screenshots/unlikely-to-buy.png" alt="Unlikely to Buy" width="850"> </p>
🎥 Demo

The Streamlit application allows users to enter different customer emails and receive an instant purchase-intent prediction.

Demo Flow
Enter Customer Email
        ↓
Click Predict
        ↓
TF-IDF Processing
        ↓
Random Forest Model
        ↓
Prediction
        ↓
Confidence Score
📦 Installation
1. Clone the repository
git clone https://github.com/madheshw265-lang/Email-Purchase-Predictor.git
2. Open the project
cd Email-Purchase-Predictor
3. Install dependencies
python -m pip install -r requirements.txt
4. Train the model
python train_model.py
5. Evaluate the model
python evaluate_model.py
6. Test the predictor
python test_predictor.py
7. Run the web application
python -m streamlit run app.py
📁 Project Structure
Email-Purchase-Predictor/
│
├── assets/
│   └── Banner.png
│
├── screenshots/
│   ├── likely-to-buy.png
│   └── unlikely-to-buy.png
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
🎯 Project Objective

The main objective of this project is to demonstrate how Natural Language Processing and Machine Learning can be used to understand customer emails and identify potential purchase intent.

Skills Demonstrated
Python
Machine Learning
Natural Language Processing
Text Classification
TF-IDF
Random Forest
Model Training
Model Evaluation
Confusion Matrix
Streamlit
Data Processing
Model Deployment Concepts
🚀 Future Improvements
📧 Gmail API integration
📥 Automatic email collection
📊 Larger real-world dataset
🤖 Transformer-based NLP
🧠 BERT-based classification
📈 Advanced analytics dashboard
☁️ Cloud deployment
🔐 User authentication
📊 Prediction history
🎯 Improved model accuracy
👨‍💻 Author
Madhesh G

Machine Learning & Software Development Enthusiast

🐙 GitHub

Username: madheshw265-lang

Profile:
https://github.com/madheshw265-lang

💼 LinkedIn

Profile:
https://www.linkedin.com/in/madheshg

⭐ Support

If you like this project, please consider giving the repository a ⭐ on GitHub.

Your support is appreciated! ❤️

📜 License

This project is created for educational and portfolio purposes.

<p align="center"> <strong>Made with ❤️ by Madhesh G</strong> </p> <p align="center"> Python • NLP • Machine Learning • Streamlit </p> ```
