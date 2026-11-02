
# RetailMind – Customer Churn Prediction

## 1. Project Overview

RetailMind is a Machine Learning project that predicts whether a telecom customer is likely to leave the company's services.

The project uses the Telco Customer Churn dataset and implements an end-to-end Machine Learning pipeline using Python and Scikit-learn.

## 2. Objectives

- Explore and understand customer data.
- Perform data cleaning and preprocessing.
- Build a reusable ML pipeline.
- Train a Logistic Regression classification model.
- Evaluate model performance.
- Save the trained pipeline using Joblib.
- Predict churn for new customers.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib

## 4. Dataset

Dataset: Telco Customer Churn

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset statistics:
- Total records: 7043
- Total columns: 21
- Churned customers: 1869
- Non-churned customers: 5174

Target column: Churn

- No = 0
- Yes = 1

## 5. Project Structure

```text
RetailMind/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── churn_pipeline.joblib
│
├── notebooks/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 6. Machine Learning Workflow

1. Load the dataset using Pandas.
2. Convert TotalCharges into numeric values.
3. Separate features and target.
4. Identify numerical and categorical columns.
5. Split the dataset into training and testing sets.
6. Handle missing numerical values using median imputation.
7. Handle missing categorical values using most-frequent imputation.
8. Scale numerical features using StandardScaler.
9. Encode categorical features using OneHotEncoder.
10. Train Logistic Regression.
11. Evaluate the trained model.
12. Save the complete pipeline using Joblib.

## 7. Model Evaluation

Algorithm: Logistic Regression

Train-test split:
- Training: 80%
- Testing: 20%

Random state: 42

Evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| ROC-AUC | 84.19% |

Confusion Matrix:

```text
[[926 109]
 [165 209]]
```

## 8. Prediction Example

The project includes a reusable prediction function in `src/predict.py`.

Example output:

```python
{
    "prediction": "Yes",
    "churn_probability": 0.6801
}
```

The model predicts that the example customer has an estimated 68.01% probability of churning.

## 9. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RetailMind
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 10. Execution

Run Exploratory Data Analysis:

```bash
python src/eda.py
```

Train the model:

```bash
python src/train.py
```

Generate predictions:

```bash
python src/predict.py
```

Note: Download the dataset from Kaggle and place it inside `data/raw/` before execution.

## 11. Future Improvements

- Experiment with Random Forest and Decision Tree.
- Improve churn recall.
- Perform hyperparameter tuning.
- Develop an API using FastAPI.
- Build a customer churn dashboard.

---

Developed as part of the LTM AI Services learning track.
