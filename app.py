
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

from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- APP ----------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Salary_Data.csv")

# ---------------- LOAD DATA ----------------

data = pd.read_csv(DATA_PATH)

required_cols = [
    "Salary",
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]

data = data.dropna(subset=required_cols)

X = data[
    [
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience"
    ]
]

y = data["Salary"]

# ---------------- PREPROCESSOR ----------------

numeric_features = [
    "Age",
    "Years of Experience"
]

categorical_features = [
    "Gender",
    "Education Level",
    "Job Title"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])

# ---------------- TRAIN MODEL ----------------

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        VotingRegressor([
            (
                "lr",
                LinearRegression()
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            ),
            (
                "gbr",
                GradientBoostingRegressor(
                    random_state=42
                )
            )
        ])
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    f"Model trained successfully | R² = "
    f"{r2_score(y_test, y_pred):.4f}"
)

# ---------------- PLOTS ----------------

def plot_to_base64(fig):
    buf = BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight"
    )

    buf.seek(0)

    image = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    buf.close()

    return image


plots = {}

sns.set_style("whitegrid")

try:

    fig, ax = plt.subplots()

    sns.histplot(
        data["Salary"],
        bins=20,
        kde=True,
        ax=ax
    )

    plots["salary_distribution"] = (
        plot_to_base64(fig)
    )

    plt.close(fig)

    fig, ax = plt.subplots()

    corr = data[
        [
            "Salary",
            "Age",
            "Years of Experience"
        ]
    ].corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    plots["correlation_heatmap"] = (
        plot_to_base64(fig)
    )

    plt.close(fig)

except Exception as e:
    print("Plot error:", e)

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        plots=plots
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        payload = request.get_json(force=True)

        df = pd.DataFrame([{
            "Age":
                float(payload["Age"]),

            "Gender":
                payload["Gender"],

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

        prediction = float(
            model.predict(df)[0]
        )

        return jsonify({
            "predicted_salary":
                round(prediction, 2)
        })

    except Exception as e:

        print(
            "Prediction Error:",
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

