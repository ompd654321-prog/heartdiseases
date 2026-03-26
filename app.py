from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

# ── Load or train model ──────────────────────────────────────────────────────
def load_or_train():
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        data = pd.read_csv("heart.csv")
        X = data.drop("target", axis=1)
        y = data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
        model.fit(X_train_s, y_train)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    return model, scaler


def get_metrics():
    data = pd.read_csv("heart.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model, _ = load_or_train()
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Downsample curve to ~30 points
    idx = np.linspace(0, len(precision) - 1, min(30, len(precision)), dtype=int)
    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": [round(float(p), 4) for p in precision[idx]],
        "recall": [round(float(r), 4) for r in recall[idx]],
    }


model, scaler = load_or_train()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    return jsonify(get_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        # User-supplied fields
        age        = float(body["age"])
        sex        = 1 if str(body["sex"]).lower() in ("1", "male", "m") else 0
        chol       = float(body["chol"])
        trestbps   = float(body["trestbps"])   # blood pressure
        thalach    = float(body["thalach"])    # max heart rate

        # Fixed / defaulted clinical fields
        cp      = int(body.get("cp",      1))
        fbs     = int(body.get("fbs",     0))
        restecg = int(body.get("restecg", 1))
        exang   = int(body.get("exang",   0))
        oldpeak = float(body.get("oldpeak", 1.0))
        slope   = int(body.get("slope",   1))
        ca      = int(body.get("ca",      0))
        thal    = int(body.get("thal",    2))

        feature_names = ["age","sex","cp","trestbps","chol",
                         "fbs","restecg","thalach","exang",
                         "oldpeak","slope","ca","thal"]

        user_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]],
            columns=feature_names,
        )

        user_scaled = scaler.transform(user_df)
        prediction  = int(model.predict(user_scaled)[0])
        probability = float(model.predict_proba(user_scaled)[0][prediction]) * 100

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 1),
            "status": "HEART DISEASE DETECTED" if prediction == 1 else "HEART DISEASE NOT DETECTED",
            "risk": "HIGH RISK" if prediction == 1 else "LOW RISK",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
