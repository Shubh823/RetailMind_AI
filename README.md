# RetailMind AI – Customer Churn Prediction Platform

An end-to-end Machine Learning platform that predicts customer churn using multiple classification algorithms, reusable preprocessing pipelines, model evaluation and model persistence.

RetailMind AI is being developed incrementally as part of the LTM AI Services learning track, with each day's implementation extending the same project and codebase.

---

## 1. Project Overview

Customer churn occurs when customers stop using a company's products or services.

For telecom companies, identifying customers who are likely to leave is important because it allows businesses to take proactive retention measures.

RetailMind AI uses historical customer information to predict whether a customer is likely to churn.

The project currently implements:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- End-to-end Machine Learning pipelines
- Multiple classification algorithms
- Model evaluation and comparison
- Model persistence using Joblib
- Reusable customer prediction functionality

---

## 2. Project Objectives

1. Analyse customer data and identify patterns associated with churn.
2. Build a reusable data preprocessing pipeline.
3. Develop a baseline Logistic Regression model.
4. Experiment with Decision Tree, Random Forest and SVM.
5. Compare models using classification metrics.
6. Select the best-performing experimental model.
7. Persist trained models for future predictions.
8. Extend the platform through hyperparameter tuning and model optimisation.

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and analysis |
| Scikit-learn | Machine Learning and preprocessing |
| Joblib | Model persistence |
| Git | Version control |
| GitHub | Source code management |

---

## 4. Dataset Information

**Dataset:** Telco Customer Churn

**Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Original dataset statistics:**

| Attribute | Value |
|---|---:|
| Total records | 7,043 |
| Total columns | 21 |
| Numerical features | 4 |
| Categorical features | 15 |
| Target column | Churn |
| Churned customers | 1,869 |
| Non-churned customers | 5,174 |

### Target Variable

The target variable is `Churn`.

- `No` → 0 (Customer stays)
- `Yes` → 1 (Customer leaves)

### Class Distribution

| Class | Customers | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |

The dataset contains an imbalance between churn and non-churn customers. Therefore, recall, precision and ROC-AUC are important evaluation metrics.

---

## 5. Project Structure

```text
RetailMind_AI/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── pipeline/
│   └── churn_analysis.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── compare_models.py
│   └── predict.py
│
├── models/
│   ├── churn_pipeline.joblib
│   ├── churn_comparison_champion.joblib
│   └── day2_model_comparison.csv
│
├── notebooks/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Day 1 – End-to-End ML Pipeline

## 6. Exploratory Data Analysis

The first increment focuses on understanding the Telco Customer Churn dataset.

The analysis includes:

- Dataset dimensions and structure
- Column data types
- Statistical summaries
- Missing value identification
- Duplicate record detection
- Target variable distribution
- Categorical feature inspection
- Data type conversion for `TotalCharges`

### Important Finding

The `TotalCharges` column was initially stored as a string.

After converting it into numeric values using Pandas, 11 missing values were identified.

These missing values are handled through median imputation in the Machine Learning pipeline.

---

## 7. Data Preprocessing

A reusable preprocessing module was created in `src/preprocessing.py`.

### Numerical Features

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

Operations:

1. Missing value imputation using `SimpleImputer(strategy="median")`.
2. Feature scaling using `StandardScaler`.

### Categorical Features

Operations:

1. Missing value imputation using `SimpleImputer(strategy="most_frequent")`.
2. Categorical encoding using `OneHotEncoder(handle_unknown="ignore")`.

### ColumnTransformer

`ColumnTransformer` combines numerical and categorical preprocessing.

The complete preprocessing workflow is integrated with the classifier using Scikit-learn's `Pipeline`.

This ensures consistent transformations during both training and prediction and prevents preprocessing data leakage.

---

## 8. Day 1 – Logistic Regression Baseline

A Logistic Regression classifier was trained as the platform's first baseline model.

### Experimental Setup

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Training data | 80% |
| Testing data | 20% |
| Training records | 5,634 |
| Testing records | 1,409 |
| Random state | 42 |
| Maximum iterations | 1,000 |
| Stratification | Enabled |

### Baseline Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| ROC-AUC | 84.19% |

### Confusion Matrix

```text
[[926 109]
 [165 209]]
```

### Baseline Model Persistence

The complete fitted preprocessing and classification pipeline is saved using Joblib.

```text
models/churn_pipeline.joblib
```

This allows the model to be loaded and reused without retraining.

---

# Day 2 – Multi-Model Comparison

## 9. Objective

The second increment extends the baseline implementation by experimenting with additional classification algorithms.

The objective is to identify a better-performing model for predicting customer churn.

### Algorithms Evaluated

1. Logistic Regression – Day 1 baseline
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)

### Experimental Setup

All models use:

- The same Telco Customer Churn dataset.
- The same 80:20 train-test split.
- The same random state of 42.
- Stratified train-test splitting.
- The same numerical and categorical preprocessing strategy.

### Algorithm Configuration

| Algorithm | Configuration |
|---|---|
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| SVM | RBF kernel, probability estimation enabled |
| Logistic Regression | `max_iter=1000` |

### Model Comparison Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 84.19% |
| Decision Tree | 72.89% | 48.96% | 50.53% | 65.73% |
| Random Forest | 77.79% | 60.34% | 47.59% | 81.62% |
| SVM | 79.13% | 64.18% | 48.40% | 79.05% |

### Day 2 Champion Selection

Decision Tree was selected as the Day 2 experimental champion among the three newly introduced algorithms.

It achieved the highest churn-class recall of 50.53%.

However, the Day 1 Logistic Regression baseline achieved a higher recall of 55.88% and ROC-AUC of 84.19%.

Therefore, Logistic Regression remains the overall best-performing model at this stage.

### Business Interpretation

The primary business objective is to identify customers who are likely to leave the telecom service.

Recall is particularly important because failing to identify a potential churner can result in customer loss.

The Day 2 experiment demonstrates that more complex algorithms do not necessarily outperform simpler baseline models.

Further model optimisation and hyperparameter tuning will be explored in the next increment.

### Day 2 Saved Artifacts

| File | Description |
|---|---|
| `churn_comparison_champion.joblib` | Decision Tree experimental champion |
| `day2_model_comparison.csv` | Performance comparison of all four models |
| `churn_pipeline.joblib` | Original Day 1 Logistic Regression baseline |

---

## 10. Prediction Service

The project provides a reusable prediction helper in `src/predict.py`.

The helper loads the persisted ML pipeline using Joblib and accepts new customer information as a Python dictionary.

### Example

```python
result = predict(customer_data)
```

Example output:

```python
{
    "prediction": "Yes",
    "churn_probability": 0.6801
}
```

The example customer has an estimated churn probability of 68.01%.

The prediction is an estimate generated by the trained model, not a guarantee of future customer behaviour.

---

## 11. Installation and Execution

### Prerequisites

- Python
- Git
- VS Code or another Python IDE

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RetailMind_AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the Telco Customer Churn dataset from Kaggle and place the CSV directly inside the `data/` folder.

### Step 5: Run Exploratory Data Analysis

```bash
python pipeline/churn_analysis.py
```

### Step 6: Test Shared Preprocessing

```bash
python src/preprocessing.py
```

### Step 7: Train Day 1 Baseline

```bash
python src/train.py
```

### Step 8: Run Day 2 Model Comparison

```bash
python src/compare_models.py
```

### Step 9: Generate Customer Predictions

```bash
python src/predict.py
```

---

## 12. Current Project Status

| Increment | Implementation | Status |
|---|---|---|
| Day 1 | Repository setup and EDA | Completed |
| Day 1 | Preprocessing and Logistic Regression | Completed |
| Day 1 | Model evaluation and Joblib persistence | Completed |
| Day 1 | Prediction helper | Completed |
| Day 2 | Shared preprocessing module | Completed |
| Day 2 | Decision Tree, Random Forest and SVM | Completed |
| Day 2 | Model comparison and champion selection | Completed |
| Day 2 | README documentation | Completed |
| Day 3 | Hyperparameter tuning | Upcoming |
| Day 3 | Bias-variance analysis | Upcoming |
| Day 3 | Learning curves and tuned champion | Upcoming |

---

## 13. Future Enhancements

- Hyperparameter optimisation using GridSearchCV.
- Cross-validation for reliable model evaluation.
- Bias and variance diagnosis.
- Learning curve analysis.
- Improved churn recall and model generalisation.
- FastAPI-based ML service.
- Interactive customer churn dashboard.

---

## 14. Author and Project Information

**Project:** RetailMind AI

**Domain:** Machine Learning / Customer Analytics

**Learning Track:** LTM AI Services

**Dataset:** Telco Customer Churn – Kaggle

This project is developed incrementally, with each implementation extending the same RetailMind AI codebase into a more complete and reusable Machine Learning platform.
