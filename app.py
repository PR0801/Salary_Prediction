import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Flask App
app = Flask(__name__)

# Load Dataset (relative path - works everywhere)
data = pd.read_csv(r"dataset/Salary_Data.csv")

# Required columns check
required_cols = ["Salary", "Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
data = data.dropna(subset=required_cols)

# Target & Features
X = data[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]
y = data["Salary"]

# Feature Groups
numeric_features = ["Age", "Years of Experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]

# Transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Ensemble Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", VotingRegressor([
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("gbr", GradientBoostingRegressor(random_state=42))
    ]))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully!")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print("Running at: http://127.0.0.1:5000")

# Convert Matplotlib figure to Base64 for HTML Rendering
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

plots = {}
sns.set_style("whitegrid")

# 1. Salary Distribution
fig, ax = plt.subplots()
sns.histplot(data['Salary'], bins=20, kde=True, ax=ax)
ax.set_title("Salary Distribution")
plots['salary_distribution'] = plot_to_base64(fig)
plt.close(fig)

# 2. Correlation Heatmap
fig, ax = plt.subplots()
corr = data[["Salary", "Age", "Years of Experience"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
plots['correlation_heatmap'] = plot_to_base64(fig)
plt.close(fig)

# 3. Salary vs Age
fig, ax = plt.subplots()
sns.scatterplot(x="Age", y="Salary", data=data, ax=ax)
ax.set_title("Salary vs Age")
plots['salary_vs_age'] = plot_to_base64(fig)
plt.close(fig)

# 4. Salary vs Experience
fig, ax = plt.subplots()
sns.scatterplot(x="Years of Experience", y="Salary", data=data, ax=ax)
ax.set_title("Salary vs Experience")
plots['salary_vs_experience'] = plot_to_base64(fig)
plt.close(fig)

# 5. Actual vs Predicted
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title("Actual vs Predicted Salaries")
plots['actual_vs_predicted'] = plot_to_base64(fig)
plt.close(fig)

# 6. Residual Distribution
fig, ax = plt.subplots()
residuals = y_test - y_pred
sns.histplot(residuals, bins=20, kde=True, ax=ax)
ax.set_title("Residuals Distribution")
plots['residuals_distribution'] = plot_to_base64(fig)
plt.close(fig)

# 7. Salary by Education Level
fig, ax = plt.subplots()
sns.boxplot(x="Education Level", y="Salary", data=data, ax=ax)
ax.set_title("Salary by Education Level")
plots['salary_by_education'] = plot_to_base64(fig)
plt.close(fig)

# 8. Average Salary by Job Title
avg_salary = data.groupby("Job Title")["Salary"].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, max(6, len(avg_salary) * 0.3)))
sns.barplot(x=avg_salary.values, y=avg_salary.index, ax=ax, palette="viridis")
ax.set_title("Average Salary by Job Title")
plots['avg_salary_by_job'] = plot_to_base64(fig)
plt.close(fig)

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html", plots=plots)

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    df = pd.DataFrame([{
        "Age": data_json.get("Age"),
        "Gender": data_json.get("Gender"),
        "Education Level": data_json.get("Education_Level"),
        "Job Title": data_json.get("Job_Title"),
        "Years of Experience": data_json.get("Years_of_Experience")
    }])
    df["Age"] = df["Age"].astype(float)
    df["Years of Experience"] = df["Years of Experience"].astype(float)
    prediction = model.predict(df)[0]
    return jsonify({"predicted_salary": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=False)
