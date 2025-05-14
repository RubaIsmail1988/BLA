import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# 1. Load dataset
df = pd.read_csv("D:/SVU/My_Clases/second/MLT/MLT_Assainment/loan_prediction.csv")

# 2. Drop rows with missing values
df = df.dropna()

# 3. Drop ID column (not useful for training)
df = df.drop("Loan_ID", axis=1)

# 4. Encode categorical features (excluding target)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("Loan_Status")  # Exclude target column
df = pd.get_dummies(df, columns=categorical_cols)

# 5. Encode target column (Loan_Status) with LabelEncoder to keep both 'Y' and 'N'
le = LabelEncoder()
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])  # 'Y' becomes 1, 'N' becomes 0

# 6. Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}
best_model = None
best_model_name = ""
best_f1 = 0

# 9. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("-" * 30)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# 10. Save the best model
joblib.dump(best_model, "model.pkl")

# 11. Save evaluation metrics to JSON
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# 12. Print best model
print(f"Best model is: {best_model_name} with F1-score = {best_f1:.4f}")