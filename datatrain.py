```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os
import joblib

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

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Salary_Data.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model.pkl"
)

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

print(f"Original rows: {len(data)}")

# ---------------- REMOVE INVALID RECORDS ----------------

data = data[
    (data["Age"] >= 18) &
    (data["Age"] <= 70)
]

data = data[
    (data["Years of Experience"] >= 0) &
    (data["Years of Experience"] <= 50)
]

data = data[
    data["Years of Experience"]
    <=
    (data["Age"] - 18)
]

# ---------------- REMOVE OUTLIERS USING IQR ----------------

def remove_outliers(df, column):

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    before = len(df)

    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]

    removed = before - len(df)

    print(
        f"{column}: Removed {removed} outliers"
    )

    return df

for col in [
    "Salary",
    "Age",
    "Years of Experience"
]:
    data = remove_outliers(data, col)

print(f"Rows after cleaning: {len(data)}")

# ---------------- FEATURES ----------------

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

# ---------------- PREPROCESSING ----------------

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
    (
        "imputer",
        SimpleImputer(strategy="mean")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
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

# ---------------- MODEL ----------------

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
                    n_estimators=200,
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

# ---------------- TRAIN ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nModel trained successfully")
print(f"R² Score: {r2:.4f}")

# ---------------- SAVE MODEL ----------------

joblib.dump(
    model,
    MODEL_FILE
)

print(f"Model saved at: {MODEL_FILE}")
```
