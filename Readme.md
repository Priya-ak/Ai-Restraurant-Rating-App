# 🍽️ AI Restaurant Rating Predictor

<img width="383" height="425" alt="image" src="https://github.com/user-attachments/assets/5fcd0a30-a3ce-4733-a1f3-19e80297fae1" />

## 🚀 Project Overview

AI Restaurant Rating Predictor is a machine learning web application that predicts the aggregate rating of a restaurant based on key business factors such as pricing, votes, and service availability.

The project covers the complete ML lifecycle:

📊 Data preprocessing

🧠 Model training

📈 Evaluation

🔍 Explainable AI (feature importance)

🌐 Deployment using Streamlit

## 🎯 Objective

Build a regression-based machine learning model to predict restaurant ratings and provide AI-driven explanations for predictions.

This project was developed as part of the Cognifyz Technologies Internship Program.

🧠 Features Used for Prediction
Feature	Description
⭐ Votes	Total customer votes

💰 Price Range	1 (Low) – 4 (High)

🍽️ Average Cost for Two	Approximate dining cost

📅 Table Booking	Yes / No

🚚 Online Delivery	Yes / No

## 📂 Project Structure
```
AI_RESTAURANT_RATING/
│
├── data/
│   └── restaurants.csv
│
├── model/
│   ├── restaurant_rating_model.pkl
│   └── importance.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

```
## ⚙️ Installation & Setup
1️⃣ Clone the Repository

```
git clone https://github.com/Priya-ak/AI_Restaurant_Rating.git
cd AI_Restaurant_Rating
```
2️⃣ Create Virtual Environment (Optional but Recommended)

```
python -m venv venv
venv\Scripts\activate
```
3️⃣ Install Dependencies

```
pip install -r requirements.txt
```
## 🧪 Model Training
Run the training script to build and save the model:
```
python train_model.py
```
✔ Saves trained model and feature importance

✔ Prints MSE and R² score
## 🌐 Run the Streamlit Web App

```
streamlit run app.py
```
📍 Open browser at:
```
http://localhost:8501
```
## 🤖 AI Explanation (Why This Rating?)

The application explains predictions using feature importance, showing:

Which factors influenced the rating most

How customer behavior and pricing affect restaurant performance

Example:
  - Votes strongly influenced the rating
  - Price range had moderate impact
  - Online delivery availability improved the score
## 📈 Model Performance

Mean Squared Error (MSE): Low (indicates accurate prediction)

R² Score: ~0.97 (strong predictive power)
## 👩‍💻 Author

Priyadharshini

🎓 Machine Learning Intern

💡 Passionate about AI & Ml

📎 Feel free to connect and explore more projects!

## ⭐ If You Like This Project

Give it a ⭐ on GitHub — it really helps! 😊

