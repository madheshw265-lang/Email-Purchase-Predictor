import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("data/emails.csv")

print("Dataset loaded successfully!")
print(f"Total emails: {len(data)}")


# ==========================================
# SEPARATE FEATURES AND LABEL
# ==========================================

X = data["email"]
y = data["bought"]


# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=1000
)

X_tfidf = vectorizer.fit_transform(X)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "model/email_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")


# ==========================================
# RESULTS
# ==========================================

print()
print("====================================")
print("       MODEL TRAINING COMPLETE")
print("====================================")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print("Model             : Random Forest")
print("Features          : TF-IDF")
print()
print("Saved files:")
print("✓ model/email_model.pkl")
print("✓ model/tfidf_vectorizer.pkl")
print("====================================")