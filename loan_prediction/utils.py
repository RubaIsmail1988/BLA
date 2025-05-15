import joblib
import numpy as np
import os

def predict_loan_status(input_data):
    
    # Load the model, scaler, and feature names from the saved .pkl file
    model_path = os.path.join("loan_prediction", "model", "lr_model.pkl")
    model_data = joblib.load(model_path)

    model = model_data['model']                  # Trained logistic regression model
    scaler = model_data['scaler']                # StandardScaler used in training
    feature_names = model_data['feature_names_in_']  # Ordered feature names used during model training

    # Construct a dictionary for the input features
    # Apply log transformation to continuous variables
    data_dict = {
        'log_ApplicantIncome': np.log1p(input_data['applicant_income']),
        'log_CoapplicantIncome': np.log1p(input_data['coapplicant_income']),
        'log_LoanAmount': np.log1p(input_data['loan_amount']),
        'Loan_Amount_Term': input_data['loan_amount_term'],
        'Credit_History': input_data['credit_history'],
        'Gender_Male': 1 if input_data['gender'] == 'Male' else 0,
        'Married_Yes': 1 if input_data['married'] == 'Yes' else 0,
        'Dependents_1': 1 if input_data['dependents'] == '1' else 0,
        'Dependents_2': 1 if input_data['dependents'] == '2' else 0,
        'Dependents_3+': 1 if input_data['dependents'] == '3+' else 0,
        'Education_Not Graduate': 1 if input_data['education'] == 'Not Graduate' else 0,
        'Self_Employed_Yes': 1 if input_data['self_employed'] == 'Yes' else 0,
        'Property_Area_Semiurban': 1 if input_data['property_area'] == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if input_data['property_area'] == 'Urban' else 0,
    }

    # If a dummy column is missing, fill it with 0
    for col in feature_names:
        if col not in data_dict:
            data_dict[col] = 0

    # Arrange the data into the correct order as expected by the model
    input_array = np.array([data_dict[col] for col in feature_names]).reshape(1, -1)

    # Scale the input using the same scaler used during training
    input_scaled = scaler.transform(input_array)

    # Make a prediction using the trained model
    prediction = model.predict(input_scaled)[0]

    # Return human-readable result
    return 'Approved' if prediction == 1 else 'Rejected'
