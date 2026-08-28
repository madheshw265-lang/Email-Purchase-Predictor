import streamlit as st

from email_predictor import predict_purchase_with_confidence


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Email Purchase Predictor",
    page_icon="📧",
    layout="centered"
)


# ==========================================
# HEADER
# ==========================================

st.title("📧 Email Purchase Intent Predictor")

st.markdown(
    "### AI-powered customer purchase intent classification"
)

st.write(
    "Analyze customer emails using Natural Language Processing "
    "(NLP), TF-IDF, and a Random Forest machine learning model."
)

st.info(
    "💡 Enter a customer email below to predict whether "
    "they are likely to purchase the product."
)

# ==========================================
# EMAIL INPUT
# ==========================================

email_text = st.text_area(
    "Enter customer email:",
    height=180,
    placeholder=(
        "Example: I am interested in your product. "
        "Can you send me the pricing and payment details?"
    )
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Purchase Intent"):

    if not email_text.strip():

        st.warning("Please enter an email first.")

    else:

        prediction, confidence = predict_purchase_with_confidence(
            email_text
        )

        st.divider()

        st.subheader("Prediction")

        if prediction == "Likely to Buy":

            st.success(f"🟢 {prediction}")

        else:

            st.error(f"🔴 {prediction}")

        st.subheader("Confidence")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(int(confidence))


# ==========================================
# MODEL INFORMATION
# ==========================================

st.divider()

st.subheader("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Algorithm**")
    st.write("Random Forest")

with col2:
    st.write("**Feature Extraction**")
    st.write("TF-IDF")


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "80%")
col2.metric("Precision", "80%")
col3.metric("Recall", "80%")
col4.metric("F1 Score", "80%")