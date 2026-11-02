
from pathlib import Path
import pandas as pd

# 1. Locate the project directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Dataset path
DATA_PATH = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 3. Load dataset
df = pd.read_csv(DATA_PATH)

# 4. Basic information
print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 5. Display first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 6. Display column information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 7. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").T)

# 8. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 9. Duplicate records
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# 10. Target variable distribution
print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

print("\n========== CHURN PERCENTAGE ==========")
print(df["Churn"].value_counts(normalize=True) * 100)

# 11. Unique values in categorical columns
print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    if df[column].dtype == "object":
        print(f"{column}: {df[column].nunique()} unique values")

# 12. Inspect TotalCharges
print("\n========== TOTAL CHARGES ==========")
print("Original dtype:", df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("New dtype:", df["TotalCharges"].dtype)
print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

print("\nEDA COMPLETED SUCCESSFULLY!")
