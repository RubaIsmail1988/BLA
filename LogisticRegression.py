import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("D:/SVU/My_Clases/second/MLT/MLT_Assainment/loan_prediction.csv")
df.drop("Loan_ID", axis=1, inplace=True)

# 2. Handle missing values
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for col in num_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
        df[col] = df.groupby(['Married', 'Education'])[col].transform(lambda x: x.fillna(x.median()))

# 3. Log transform to reduce skewness
for col in ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])

# 4. One-hot encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')  # Remove target column from the list
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 5. Encode target
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# 6. Features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# 7. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 9. Handle imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 10. Train model
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='liblinear',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# 11. Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=2))

# 12. Save model
os.makedirs("loan_prediction/model", exist_ok=True)
joblib.dump({
    'model': model,
    'scaler': scaler,
    'feature_names': X.columns.tolist()
}, "loan_prediction/model/lr_model.pkl")

# 13. Save metrics
with open("lr_metrics.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report
    }, f, indent=4)

print("LogisticRegression model and metrics saved successfully.")

# Load the saved model and scaler
loaded_model = joblib.load("loan_prediction/model/lr_model.pkl")

# Predict
y_pred = loaded_model['model'].predict(loaded_model['scaler'].transform(X))
print("Predictions on the original scale:", y_pred)

# 14. Visualize results
plt.figure(figsize=(10, 5))

# Plot: Loan Amount vs Applicant Income
plt.subplot(1, 2, 1)
plt.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.6)
plt.xlabel('Applicant Income (Capped)')
plt.ylabel('Loan Amount (Capped)')
plt.title('Applicant Income vs Loan Amount')
plt.grid(True)

# Plot: Boxplot of log_LoanAmount by Loan Status
plt.subplot(1, 2, 2)
sns.boxplot(x='Loan_Status', y='log_LoanAmount', data=df)
plt.xlabel('Loan Status (0=Rejected, 1=Approved)')
plt.ylabel('Log Loan Amount')
plt.title('Loan Amount by Loan Status')

plt.tight_layout()
plt.show()
