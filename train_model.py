import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data/restaurants.csv")

# -------------------------------
# Convert Yes/No to 1/0
# -------------------------------
yes_no_cols = [
    "Has Table booking",
    "Has Online delivery"
]

for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# -------------------------------
# SELECT ONLY FEATURES WE USE IN APP
# -------------------------------
features = [
    "Votes",
    "Price range",
    "Average Cost for two",
    "Has Table booking",
    "Has Online delivery"
]

X = df[features]
y = df["Aggregate rating"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# Feature Importance
# -------------------------------
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("model", exist_ok=True)

with open("model/restaurant_rating_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/importance.pkl", "wb") as f:
    pickle.dump(importance_df, f)

print("✅ Model retrained and saved successfully")
