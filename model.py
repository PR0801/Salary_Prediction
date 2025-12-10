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
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DATASET", "Salary_Data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

data = pd.read_csv(DATA_PATH)
required_cols = ["Salary","Age","Gender","Education Level","Job Title","Years of Experience"]
data = data.dropna(subset=required_cols)

X = data[["Age","Gender","Education Level","Job Title","Years of Experience"]]
y = data["Salary"]

numeric_features = ["Age","Years of Experience"]
categorical_features = ["Gender","Education Level","Job Title"]

numeric_transformer = Pipeline([("imputer",SimpleImputer(strategy="mean")),("scaler",StandardScaler())])
categorical_transformer = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

preprocessor = ColumnTransformer([("num",numeric_transformer,numeric_features),
                                  ("cat",categorical_transformer,categorical_features)])

if not os.path.exists(MODEL_FILE):
    model = Pipeline([
        ("preprocessor",preprocessor),
        ("regressor",VotingRegressor([
            ("lr",LinearRegression()),
            ("rf",RandomForestRegressor(n_estimators=200,random_state=42)),
            ("gbr",GradientBoostingRegressor(random_state=42))
        ]))
    ])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    print("Model trained successfully!")
    print(f"R² Score: {r2:.2f}")
    joblib.dump(model,MODEL_FILE)
else:
    model = joblib.load(MODEL_FILE)

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf,format="png",bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

plots = {}
sns.set_style("whitegrid")

fig,ax = plt.subplots()
sns.histplot(data['Salary'],bins=20,kde=True,ax=ax)
plots['salary_distribution'] = plot_to_base64(fig)
plt.close(fig)

fig,ax = plt.subplots()
corr = data[["Salary","Age","Years of Experience"]].corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap="coolwarm",ax=ax)
plots['correlation_heatmap'] = plot_to_base64(fig)
plt.close(fig)

fig,ax = plt.subplots()
sns.scatterplot(x="Age",y="Salary",data=data,ax=ax)
plots['salary_vs_age'] = plot_to_base64(fig)
plt.close(fig)

fig,ax = plt.subplots()
sns.scatterplot(x="Years of Experience",y="Salary",data=data,ax=ax)
plots['salary_vs_experience'] = plot_to_base64(fig)
plt.close(fig)

fig,ax = plt.subplots()
sns.scatterplot(x=y,y=model.predict(X),ax=ax)
plots['actual_vs_predicted'] = plot_to_base64(fig)
plt.close(fig)

fig,ax = plt.subplots()
residuals = y - model.predict(X)
sns.histplot(residuals,bins=20,kde=True,ax=ax)
plots['residuals_distribution'] = plot_to_base64(fig)
plt.close(fig)

fig,ax = plt.subplots()
sns.boxplot(x="Education Level",y="Salary",data=data,ax=ax)
plots['salary_by_education'] = plot_to_base64(fig)
plt.close(fig)

avg_salary = data.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(25)
fig,ax = plt.subplots(figsize=(10,max(6,len(avg_salary)*0.4)))
sns.barplot(x=avg_salary.values,y=avg_salary.index,ax=ax,palette="viridis")
plots['avg_salary_by_job'] = plot_to_base64(fig)
plt.close(fig)

@app.route("/")
def home():
    return render_template("index.html",plots=plots)

@app.route("/predict",methods=["POST"])
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
        return jsonify({"predicted_salary": round(prediction,2)})
    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__ == "__main__":
    print("Running at: http://127.0.0.1:5000")
    app.run(debug=False)
