import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
import joblib
import os

# ---------------- APP ----------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Salary_Data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

# ---------------- LOAD MODEL & DATA ----------------
model = joblib.load(MODEL_FILE)
data = pd.read_csv(DATA_PATH)

# ---------------- UTIL ----------------
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

plots = {}
sns.set_style("whitegrid")

# ---------------- PLOTS ----------------
fig, ax = plt.subplots()
sns.histplot(data["Salary"], bins=20, kde=True, ax=ax)
plots["salary_distribution"] = plot_to_base64(fig)
plt.close(fig)

fig, ax = plt.subplots()
corr = data[["Salary", "Age", "Years of Experience"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
plots["correlation_heatmap"] = plot_to_base64(fig)
plt.close(fig)

fig, ax = plt.subplots()
sns.scatterplot(x="Age", y="Salary", data=data, ax=ax)
plots["salary_vs_age"] = plot_to_base64(fig)
plt.close(fig)

fig, ax = plt.subplots()
sns.scatterplot(x="Years of Experience", y="Salary", data=data, ax=ax)
plots["salary_vs_experience"] = plot_to_base64(fig)
plt.close(fig)

fig, ax = plt.subplots()
sns.scatterplot(x=data["Salary"], y=model.predict(
    data[["Age","Gender","Education Level","Job Title","Years of Experience"]]
), ax=ax)
plots["actual_vs_predicted"] = plot_to_base64(fig)
plt.close(fig)

fig, ax = plt.subplots()
residuals = data["Salary"] - model.predict(
    data[["Age","Gender","Education Level","Job Title","Years of Experience"]]
)
sns.histplot(residuals, bins=20, kde=True, ax=ax)
plots["residuals_distribution"] = plot_to_base64(fig)
plt.close(fig)

fig, ax = plt.subplots()
sns.boxplot(x="Education Level", y="Salary", data=data, ax=ax)
plots["salary_by_education"] = plot_to_base64(fig)
plt.close(fig)

avg_salary = (
    data.groupby("Job Title")["Salary"]
    .mean()
    .sort_values(ascending=False)
    .head(25)
)

fig, ax = plt.subplots(figsize=(10, max(6, len(avg_salary) * 0.4)))
sns.barplot(x=avg_salary.values, y=avg_salary.index, ax=ax, palette="viridis")
plots["avg_salary_by_job"] = plot_to_base64(fig)
plt.close(fig)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html", plots=plots)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_json = request.get_json()

        df = pd.DataFrame([{
            "Age": float(data_json.get("Age")),
            "Gender": data_json.get("Gender"),
            "Education Level": data_json.get("Education_Level"),
            "Job Title": data_json.get("Job_Title"),
            "Years of Experience": float(data_json.get("Years_of_Experience"))
        }])

        prediction = model.predict(df)[0]
        return jsonify({"predicted_salary": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🚀 Running at: http://127.0.0.1:5000")
    app.run(debug=False)
