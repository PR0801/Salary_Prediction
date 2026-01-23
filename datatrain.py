import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset/Salary_Data.csv")

X = data[["YearsExperience"]]
y = data["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
