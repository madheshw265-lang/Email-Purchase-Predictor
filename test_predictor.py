from email_predictor import predict_purchase


emails = [
    "I am interested in buying your product. Please send me the price.",
    "Can I place an order today?",
    "Please send me the payment details.",
    "I am not interested in this product.",
    "Please stop sending me emails.",
    "I don't want to buy anything."
]


for email in emails:

    prediction = predict_purchase(email)

    print("Email:", email)
    print("Prediction:", prediction)
    print("-" * 50)