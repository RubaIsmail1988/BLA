import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("BLA/loan_prediction/dataset/loan_prediction.csv")
df.drop("Loan_ID", axis=1, inplace=True)

# 2. Handle missing values
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical nulls with mode

num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for col in num_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)  # Cap outliers
        df[col] = df.groupby(['Married', 'Education'])[col].transform(lambda x: x.fillna(x.median()))  # Fill nulls with median by group

# 3. Convert '3+' to integer in Dependents
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = df['Dependents'].astype(int)


# 4. One-hot encode categorical features (except target)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')

# Use get_dummies with drop_first=True to avoid multicollinearity
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 5. Encode target variable
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# 6. Define feature columns explicitly to fix order and names
feature_names = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Gender_Male",
    "Married_Yes",
    "Dependents_1",
    "Dependents_2",
    "Dependents_3",
    "Education_Not Graduate",
    "Self_Employed_Yes",
    "Property_Area_Semiurban",
    "Property_Area_Urban"
]

# 7. Confirm all features exist in dataframe (if some missing, add zero column)
for feat in feature_names:
    if feat not in df.columns:
        print(f"Feature '{feat}' not found in dataframe, adding column with zeros.")
        df[feat] = 0  # add missing columns as zeros for consistency

# 8. Extract features and target in fixed order
X = df[feature_names]
y = df["Loan_Status"]

# 9. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 11. Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 12. Train logistic regression
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='liblinear',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# 13. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=2))

# 14. Save evaluation metrics to JSON
model_dir = "BLA/loan_prediction/model"
os.makedirs(model_dir, exist_ok=True)

metrics = {
    "accuracy": accuracy,
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": {}
}

for label, values in report.items():
    if isinstance(values, dict):
        metrics["classification_report"][label] = {
            "precision": values.get("precision", 0),
            "recall": values.get("recall", 0),
            "f1_score": values.get("f1-score", 0),
            "support": values.get("support", 0)
        }

with open(os.path.join(model_dir, "lr_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation metrics saved to lr_metrics.json.")

# 15. Save model parameters to JSON with fixed feature_names order
model_data = {
    "coefficients": model.coef_[0].tolist(),
    "intercept": model.intercept_[0],
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "feature_names": feature_names
}

with open(os.path.join(model_dir, "model_params.json"), "w") as f:
    json.dump(model_data, f, indent=4)

print("Model parameters saved successfully as JSON.")

# Visualization (optional, unchanged)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.6)
plt.xlabel('Applicant Income (Capped)')
plt.ylabel('Loan Amount (Capped)')
plt.title('Applicant Income vs Loan Amount')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.boxplot(x='Loan_Status', y='LoanAmount', data=df)
plt.xlabel('Loan Status (0=Rejected, 1=Approved)')
plt.ylabel('Loan Amount')
plt.title('Loan Amount by Loan Status')

plt.tight_layout()
plt.show()
