import warnings
warnings.filterwarnings("ignore")

import os
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Salary_Data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_FILE)

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return img

def generate_plots():
    plots = {}
    sns.set_style("whitegrid")

    fig, ax = plt.subplots()
    sns.histplot(data["Salary"], bins=20, kde=True, ax=ax)
    plots["salary_distribution"] = plot_to_base64(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    corr = data[["Salary", "Age", "Years of Experience"]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plots["correlation_heatmap"] = plot_to_base64(fig)
    plt.close(fig)

    return plots

@app.route("/")
def home():
    plots = generate_plots()
    return render_template("index.html", plots=plots)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_json = request.get_json()

        df = pd.DataFrame([{
            "Age": float(data_json["Age"]),
            "Gender": data_json["Gender"],
            "Education Level": data_json["Education_Level"],
            "Job Title": data_json["Job_Title"],
            "Years of Experience": float(data_json["Years_of_Experience"])
        }])

        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_salary": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
