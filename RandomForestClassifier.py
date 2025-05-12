import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("D:/SVU/My_Clases/second/MLT/MLT_Assainment/loan_prediction.csv")

# 2. Drop Loan_ID column if it exists
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

# 3. Fill missing values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

# 4. Remove outliers using IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers_iqr(df, 'ApplicantIncome')
df = remove_outliers_iqr(df, 'LoanAmount')

# 5. One-hot encode categorical features (excluding the target)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Loan_Status" in categorical_cols:
    categorical_cols.remove("Loan_Status")

df = pd.get_dummies(df, columns=categorical_cols)

# 6. Encode the target column
le = LabelEncoder()
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])  # Y → 1, N → 0

# 7. Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# 8. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 10. Make predictions and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=2))

# 11. Save the model
joblib.dump(rf, "rf_model.pkl")

# 12. Save metrics to JSON file
metrics_dict = {
    "accuracy": accuracy,
    "classification_report": classification_report(y_test, y_pred, digits=2, output_dict=True)
}
with open("rf_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("Random Forest model and metrics saved successfully.")

# 13. Create new Gender_Label column from encoded Gender
df['Gender_Label'] = df.apply(
    lambda row: 'Male' if row.get('Gender_Male', 0) == 1 else 'Female', axis=1
)

# 14. Visualizations side by side
plt.figure(figsize=(12, 6))

# Plot 1: Scatter plot for ApplicantIncome vs LoanAmount
plt.subplot(1, 2, 1)
plt.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.6)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Relation: Applicant Income vs Loan Amount')
plt.grid(True)

# Plot 2: Boxplot for ApplicantIncome by Gender
plt.subplot(1, 2, 2)
sns.boxplot(x='Gender_Label', y='ApplicantIncome', data=df)
plt.xlabel('Gender')
plt.ylabel('Applicant Income')
plt.title('Income Distribution by Gender')
plt.grid(True)

# Final layout
plt.tight_layout()
plt.show()
