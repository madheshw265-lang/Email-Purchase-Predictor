import re
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("model/email_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):
    """
    Clean email text before prediction.
    """

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# PURCHASE PREDICTION
# ==========================================

def predict_purchase(email_text):
    """
    Predict whether an email indicates
    purchase intent.
    """

    cleaned_email = clean_text(email_text)

    email_vector = vectorizer.transform([cleaned_email])

    prediction = model.predict(email_vector)[0]

    if prediction == 1:
        return "Likely to Buy"

    return "Unlikely to Buy"


# ==========================================
# PREDICTION + CONFIDENCE
# ==========================================

def predict_purchase_with_confidence(email_text):
    """
    Return prediction and model confidence.
    """

    cleaned_email = clean_text(email_text)

    email_vector = vectorizer.transform([cleaned_email])

    prediction = model.predict(email_vector)[0]

    probabilities = model.predict_proba(email_vector)[0]

    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "Likely to Buy"
    else:
        result = "Unlikely to Buy"

    return result, confidence