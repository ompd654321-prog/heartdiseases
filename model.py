import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
import os

# Load dataset
data = pd.read_csv("heart.csv")

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Precision-Recall curve
y_prob = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)

print(f"Model trained. Accuracy: {round(accuracy * 100, 2)}%")
print("Model and scaler saved.")
