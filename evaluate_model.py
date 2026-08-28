import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. TEXT CLEANING
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================
# 2. DATASET
# ==========================================

data = {
    "email": [

        # -------- POSITIVE --------

        "I want to buy this product",
        "Can you send me the price?",
        "How much does this product cost?",
        "I am interested in purchasing",
        "Can I place an order?",
        "Please send payment details",
        "I would like to purchase this",
        "Is the product available?",
        "Can you tell me about the pricing?",
        "I want to order two units",
        "I would like to buy this product",
        "Please tell me how to place an order",
        "Can I purchase this today?",
        "I am ready to buy",
        "I would like to know the payment options",
        "Please send me the checkout link",
        "How can I order this product?",
        "I want to purchase two products",
        "I am interested in buying this",
        "Please provide the price and payment details",

        # -------- NEGATIVE --------

        "I am not interested",
        "Please remove me from my mailing list",
        "I don't want this product",
        "Not interested at this time",
        "I am just browsing",
        "I don't need this",
        "Please stop sending emails",
        "I will not purchase this",
        "No thanks",
        "I am not looking to buy",
        "I don't want to buy anything",
        "I have no interest in purchasing",
        "I am not interested in buying",
        "I don't need this product",
        "I don't want to purchase anything",
        "Please do not send me product offers",
        "I am not planning to buy",
        "I won't be buying this",
        "I don't want to place an order",
        "I am only looking, not buying"
    ],

    "bought": [

        # Positive
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,

        # Negative
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    ]
}


# ==========================================
# 3. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(data)

df["email"] = df["email"].apply(clean_text)

X = df["email"]
y = df["bought"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 6. RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 7. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n====================================")
print("       MODEL EVALUATION")
print("====================================")

print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n====================================")
print("       CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Unlikely to Buy",
            "Likely to Buy"
        ],
        zero_division=0
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\n====================================")
print("       CONFUSION MATRIX")
print("====================================")

cm = confusion_matrix(y_test, y_pred)

print(cm)