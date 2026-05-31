import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Salary_Data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

# ---------------- LOAD ----------------

try:

    model = joblib.load(MODEL_PATH)

    print("✅ Model Loaded")

except Exception as e:

    print("❌ MODEL ERROR:", e)

    raise

try:

    dataset = pd.read_csv(DATA_PATH)

    print("✅ Dataset Loaded")

except Exception as e:

    print("❌ DATA ERROR:", e)

    raise


# ---------------- HOME ----------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        plots={}
    )


# ---------------- HEALTH ----------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        payload = request.get_json(force=True)

        print("Incoming:", payload)

        df = pd.DataFrame([{
            "Age": float(payload["Age"]),
            "Gender": payload["Gender"],
            "Education Level":
                payload["Education_Level"],
            "Job Title":
                payload["Job_Title"],
            "Years of Experience":
                float(
                    payload[
                        "Years_of_Experience"
                    ]
                )
        }])

        prediction = model.predict(df)

        salary = float(prediction[0])

        return jsonify({
            "predicted_salary": salary
        })

    except Exception as e:

        print(
            "❌ PREDICTION ERROR:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ---------------- MAIN ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
