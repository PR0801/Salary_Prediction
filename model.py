import warnings
warnings.filterwarnings("ignore")  # suppress all warnings

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
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Flask app
app = Flask(__name__)

# Load dataset (use raw string for file path)
data = pd.read_csv(r"IBM_PBEL\Project-2\DATASET\Salary_Data.csv")

# Drop rows with missing target or features
required_cols = ["Salary", "Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
data = data.dropna(subset=required_cols)

# Features and target
X = data[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]
y = data["Salary"]

# Numeric and categorical features
numeric_features = ["Age", "Years of Experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]

# Imputers for missing values handling
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Numeric pipeline: imputer + scaler
numeric_transformer = Pipeline(steps=[
    ('imputer', numeric_imputer),
    ('scaler', StandardScaler())
])

# Categorical pipeline: imputer + one-hot encoder
categorical_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model pipeline with preprocessor + voting regressor
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", VotingRegressor([
            ("lr", LinearRegression()),
            ("dt", DecisionTreeRegressor())
        ]))
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model trained successfully! MSE={mse:.2f}, R2={r2:.2f}")
print(" * Running on http://127.0.0.1:5000")

# Encode plots for HTML display
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

plots = {}
sns.set_style("whitegrid")

# Salary Distribution Plot
fig, ax = plt.subplots()
sns.histplot(data['Salary'], bins=20, kde=True, ax=ax)
ax.set_title("Salary Distribution")
plots['salary_distribution'] = plot_to_base64(fig)
plt.close(fig)

# Correlation Heatmap
fig, ax = plt.subplots()
corr = data[["Salary", "Age", "Years of Experience"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
plots['correlation_heatmap'] = plot_to_base64(fig)
plt.close(fig)

# Salary vs Age Scatterplot
fig, ax = plt.subplots()
sns.scatterplot(x="Age", y="Salary", data=data, ax=ax)
ax.set_title("Salary vs Age")
plots['salary_vs_age'] = plot_to_base64(fig)
plt.close(fig)

# Salary vs Experience Plot
fig, ax = plt.subplots()
sns.scatterplot(x="Years of Experience", y="Salary", data=data, ax=ax)
ax.set_title("Salary vs Experience")
plots['salary_vs_experience'] = plot_to_base64(fig)
plt.close(fig)

# Actual vs Predicted Salaries
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title("Actual vs Predicted Salaries")
plots['actual_vs_predicted'] = plot_to_base64(fig)
plt.close(fig)

# Residuals Distribution
fig, ax = plt.subplots()
residuals = y_test - y_pred
sns.histplot(residuals, bins=20, kde=True, ax=ax)
ax.set_title("Residuals Distribution")
plots['residuals_distribution'] = plot_to_base64(fig)
plt.close(fig)

# Salary by Education Level Boxplot
fig, ax = plt.subplots()
sns.boxplot(x="Education Level", y="Salary", data=data, palette="Set2", hue=None, ax=ax)
ax.set_title("Salary by Education Level")
plots['salary_by_education'] = plot_to_base64(fig)
plt.close(fig)

# Average Salary by Job Title Barplot (improved for clarity)
avg_salary = data.groupby("Job Title")["Salary"].mean().sort_values()
topN = 15 if len(avg_salary) > 15 else len(avg_salary)
avg_salary_top = avg_salary.tail(topN)
fig, ax = plt.subplots(figsize=(10, max(6, topN * 0.5)))
sns.barplot(x=avg_salary_top.values, y=avg_salary_top.index, palette="viridis", hue=None, ax=ax)
ax.set_title("Top Job Titles by Average Salary")
ax.tick_params(axis='y', labelsize=10)
fig.tight_layout()
plots['avg_salary_by_job'] = plot_to_base64(fig)
plt.close(fig)

# Flask routes
@app.route("/")
def home():
    return render_template("index.html", plots=plots)

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    try:
        # Map HTML form fields to model's expected column names
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
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False)
