---
# Salary Prediction Model

Predict employee salaries using several features (age, gender, years of experience, job title, education level, location etc) with machine learning (including ensemble methods).

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Licensing & Attribution](#licensing--attribution)


---

## Project Overview

This project aims to build a machine learning pipeline that uses multiple features about an employee (such as age, gender, education level, job title, years of experience, location etc) to **predict the salary**.
It applies data preprocessing, exploratory data analysis (EDA), visualisations, model training using different algorithms (including ensemble techniques like Random Forest, Gradient Boosting, Voting Regressor) and deployment via a simple web interface (Flask).

## Problem Statement

Given an employee’s information (age, gender, education level, job title, years of experience, etc.), predict their salary. The objective is to build accurate regression models and then optimise performance via ensemble methods, yielding better generalisation and robustness.

## Dataset

* Kaggle Link-https://www.kaggle.com/datasets/mubeenshehzadi/salary-dataset
* The dataset used is derived from a public source (e.g., Kaggle) containing salary data along with relevant features.
* Example features include: Salary (target), Age, Gender, Education Level, Job Title, Years of Experience (and optionally Location).
* Rows with missing values in required columns are dropped to maintain clean data for modelling.
* Features are both numeric (e.g., Age, Years of Experience) and categorical (Gender, Education Level, Job Title).

## Features

### Numeric features

* `Age`
* `Years of Experience`

### Categorical features

* `Gender`
* `Education Level`
* `Job Title`

Target variable: `Salary`

## Methodology

1. **Data loading & cleaning**

   * Load dataset from CSV
   * Drop rows with missing values in required columns
2. **Exploratory Data Analysis & Visualisation**

   * Salary distribution
   * Correlation heatmap of numeric features
   * Scatter plots (Salary vs Age, Salary vs Years of Experience)
   * Boxplot of Salary by Education Level
   * Bar chart of average Salary by Job Title
   * Residual distribution
   * Actual vs Predicted Salary plot
3. **Preprocessing Pipeline**

   * Numeric transformer: impute missing (mean) + standard scale
   * Categorical transformer: impute missing (most frequent) + one-hot encode
   * Combined into `ColumnTransformer`
4. **Modeling**

   * Use a `Pipeline` combining preprocessing + regressor
   * Regressor: `VotingRegressor` combining:

     * `LinearRegression`
     * `RandomForestRegressor`
     * `GradientBoostingRegressor`
   * Split data into train/test (e.g., 80/20)
   * Fit model on training set
5. **Evaluation**

   * Predict on test set
   * Compute metrics: Mean Squared Error (MSE), R² Score
   * Visualise predictions vs actuals, residuals
6. **Web App Integration (Flask)**

   * Build a simple Flask web interface where user inputs Age, Gender, Education Level, Job Title, Years of Experience
   * Prediction endpoint returns predicted salary in JSON
   * Render visualisations in the web interface by converting matplotlib figures to Base64 images

## Evaluation

* The model prints out the MSE and R² score following test predictions.
* Visualisations provide insight into feature-target relationships and model residuals/fit.
* Ensemble approach (VotingRegressor of Linear + RF + GB) improves prediction robustness over a single algorithm.

## Usage

### Prerequisites

* Python 3.x
* Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `flask`
* Install via `pip install -r requirements.txt`

### Steps

1. Clone/download the repository.
2. Place the dataset (`Salary_Data.csv`) under the `DATASET/` folder (or update the path accordingly).
3. Adjust file paths in the script if needed.
4. Run the main script (e.g., `python model.py` or `python app.py`).

   * The model will train, print evaluation metrics, and start the Flask web server (e.g., at `http://127.0.0.1:5000`).
5. Navigate to the home page to see the visualisations and use the prediction form.
6. For prediction via API: send a POST request to `/predict` endpoint with JSON payload:

```json
{
  "Age": <float>,
  "Gender": "<string>",
  "Education_Level": "<string>",
  "Job_Title": "<string>",
  "Years_of_Experience": <float>
}
```

Response will include:

```json
{"predicted_salary": <float>}
```

## Project Structure

```
Salary_Prediction/
│
├── DATASET/
│   └── Salary_Data.csv
├── template/
│   └── index.html
├── model.py            # main training + web app script
├── requirements.txt    # list of dependencies
├── README.md           # this file
└── … (other files) …
```

## Future Enhancements

* **Feature expansion**: Add more features (location, company size, job skills, industry sector, certifications) to improve accuracy.
* **Hyperparameter tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimise model parameters for RF / GB.
* **Model explainability**: Integrate SHAP or LIME to interpret feature importance.
* **Deployment**: Deploy the Flask app to cloud (Heroku, AWS, GCP) for public access.
* **User interface**: Improve UI/UX (use Streamlit or React) for better prediction workflow and interactive visualisations.
* **Model versioning & monitoring**: Track model performance over time, log metrics, handle data drift.
* **Additional visualisations**: Interactive plots (Plotly) for web app.

## Licensing & Attribution

This project is open source—you may use and adapt it for educational & non-commercial purposes.
If you derived your dataset from Kaggle or another source, please cite the original dataset and respect its licensing terms.

---

Thank you for checking out this project! Feedback and contributions are welcome.

---
