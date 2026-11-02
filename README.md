# 12. Inspect TotalCharges
print("\n========== TOTAL CHARGES ==========")
print("Original dtype:", df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
    )

    print("New dtype:", df["TotalCharges"].dtype)
    print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

    print("\nEDA COMPLETED SUCCESSFULLY!")