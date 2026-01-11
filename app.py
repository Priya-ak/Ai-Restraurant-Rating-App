import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load Model & Importance
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("model/restaurant_rating_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/importance.pkl", "rb") as f:
        importance_df = pickle.load(f)

    return model, importance_df


model, importance_df = load_artifacts()

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="AI Restaurant Rating Predictor", layout="centered")

st.title("🍽️ AI Restaurant Rating Predictor")
st.write("Predict restaurant ratings and understand **why** the AI made that decision.")

st.divider()

# -------------------------------
# Input Form (Simplified & Safe)
# -------------------------------
st.subheader("Enter Restaurant Details")

votes = st.number_input("Votes", min_value=0, value=50)
price_range = st.selectbox("Price Range (1 = Low, 4 = High)", [1, 2, 3, 4])
avg_cost = st.number_input("Average Cost for Two", min_value=50, value=500)

table_booking = st.selectbox("Has Table Booking?", ["Yes", "No"])
online_delivery = st.selectbox("Has Online Delivery?", ["Yes", "No"])

# Convert Yes/No to 1/0
table_booking = 1 if table_booking == "Yes" else 0
online_delivery = 1 if online_delivery == "Yes" else 0

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame([{
    "Votes": votes,
    "Price range": price_range,
    "Average Cost for two": avg_cost,
    "Has Table booking": table_booking,
    "Has Online delivery": online_delivery
}])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Rating"):
    prediction = model.predict(input_data)[0]
    prediction = round(float(prediction), 1)

    st.success(f"⭐ Predicted Restaurant Rating: **{prediction} / 5**")

    # -------------------------------
    # AI Explanation
    # -------------------------------
    st.subheader("🤖 AI Explanation (Why this rating?)")

    top_features = importance_df.head(3)

    for _, row in top_features.iterrows():
        feature = row["feature"]
        if feature in input_data.columns:
            value = input_data[feature].values[0]
            st.write(f"• **{feature}** strongly influenced the rating (value: {value})")

    st.caption("Explanation based on feature importance from the trained ML model.")
