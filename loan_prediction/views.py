import os
import json
import joblib
import base64
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import LoanRequestForm
from .models import LoanRequest


# Load model, scaler, and feature names
model = None
scaler = None
feature_names = None

try:
    model_path = os.path.join(settings.BASE_DIR, 'loan_prediction', 'model', 'lr_model.pkl')
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)
        model = model_data['model']
        scaler = model_data['scaler']

    features_path = os.path.join(settings.BASE_DIR, 'loan_prediction', 'model', 'feature_names.pkl')
    feature_names = joblib.load(features_path)

except Exception as e:
    print(f"Error loading model, scaler, or feature names: {e}")

# Home Page View
def home(request):
    return render(request, 'pages/home.html')

# Prediction view
with open("BLA/loan_prediction/model/model_params.json", "r") as f:
    params = json.load(f)

# Extract logistic regression coefficients, intercept, and scaler params
coefficients = np.array(params["coefficients"])
intercept = params["intercept"]
mean = np.array(params["scaler_mean"])
scale = np.array(params["scaler_scale"])

def standard_scale(X):
    """
    Apply standard scaling: (X - mean) / scale
    """
    return (X - mean) / scale

def logistic_sigmoid(x):
    """
    Compute the sigmoid function safely for scalars and arrays.
    """
    x = np.asarray(x, dtype=np.float64)  # Ensure input is a NumPy array of floats
    return 1 / (1 + np.exp(-x))

def predict_proba(X_raw):
    """
    Predict probability of positive class using
    logistic regression parameters and manual scaling.
    X_raw: numpy array of shape (1, n_features)
    Returns: probability scalar
    """
    X_scaled = standard_scale(X_raw)
    linear_output = np.dot(X_scaled, coefficients) + intercept
    prob = logistic_sigmoid(linear_output)
    return prob[0]  # Return scalar probability

def predict_view(request):
    """
    View to handle loan prediction requests.
    Keeps same interface as before: form, result, probability.
    """
    result = None
    probability = None

    if request.method == 'POST':
        form = LoanRequestForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Construct feature array in same order as model training
            input_features = np.array([
                data['applicant_income'],
                data['coapplicant_income'],
                data['loan_amount'],
                np.log1p(data['loan_amount']),
                np.log1p(data['applicant_income']),
                np.log1p(data['coapplicant_income']),
                float(data['credit_history']),
                int(data['gender'] == 'Male'),
                int(data['married'] == 'Yes'),
                int(data['education'] == 'Graduate'),
                int(data['self_employed'] == 'Yes'),
                int(data['property_area'] == 'Semiurban'),
                int(data['property_area'] == 'Urban'),
                int(data['dependents'] == '1'),
                int(data['dependents'] == '2'),
                int(data['dependents'] in ['3', '3']),
                int(data['loan_amount_term'] == 360.0)
            ]).reshape(1, -1)

            # Predict probability using manual logistic regression implementation
            prob = predict_proba(input_features)
            probability = round(prob * 100, 2)

            # Define decision thresholds for result label
            if prob >= 0.7:
                result = "Approved"
            elif prob >= 0.5:
                result = "Under Review"
            else:
                result = "Rejected"

            # Save form instance with predicted loan status
            loan_request = form.save(commit=False)
            loan_request.loan_status = result
            loan_request.save()
    else:
        form = LoanRequestForm()

    # Render template with form, result, and probability
    return render(request, 'pages/predict.html', {
        'form': form,
        'result': result,
        'probability': probability
    })

# Loan Requests Management View
def manage_requests_view(request):
    if request.method == 'POST':
        req_id = request.POST.get('request_id')
        obj = get_object_or_404(LoanRequest, pk=req_id)
        obj.delete()
        return redirect('loan_prediction:requests')

    requests = LoanRequest.objects.all()
    return render(request, 'pages/requests.html', {'requests': requests})
    

def eda_view(request):
    plots = {}
    data_table = []

    try:
        # Load all loan requests from the database
        queryset = LoanRequest.objects.all()

        # Convert queryset to DataFrame
        df = pd.DataFrame(list(queryset.values()))

        if df.empty:
            raise ValueError("No loan requests in the database.")

        # Convert categorical fields to appropriate type
        df['Credit_History'] = df['credit_history'].fillna(0)
        df['LoanAmount'] = df['loan_amount']
        df['ApplicantIncome'] = df['applicant_income']
        df['Loan_Status'] = df['loan_status']
        df['Property_Area'] = df['property_area']

        # Plot 1: Applicant Income vs Loan Amount
        fig1, ax1 = plt.subplots()
        ax1.scatter(df['ApplicantIncome'], df['LoanAmount'], alpha=0.5)
        ax1.set_xlabel('Applicant Income')
        ax1.set_ylabel('Loan Amount')
        ax1.set_title('Applicant Income vs Loan Amount')
        buf1 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf1, format='png')
        plt.close(fig1)
        plots['Income vs Loan'] = base64.b64encode(buf1.getvalue()).decode('utf-8')

        # Plot 2: Boxplot of Loan Amount by Loan Status
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Loan_Status', y='LoanAmount', data=df, ax=ax2)
        ax2.set_title('Loan Amount Distribution by Status')
        buf2 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png')
        plt.close(fig2)
        plots['Loan Amount by Status'] = base64.b64encode(buf2.getvalue()).decode('utf-8')

        # Plot 3: Countplot of Credit History
        fig3, ax3 = plt.subplots()
        sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=ax3)
        ax3.set_title('Loan Status by Credit History')
        buf3 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf3, format='png')
        plt.close(fig3)
        plots['Credit History'] = base64.b64encode(buf3.getvalue()).decode('utf-8')

        # Plot 4: Countplot of Property Area
        fig4, ax4 = plt.subplots()
        sns.countplot(x='Property_Area', hue='Loan_Status', data=df, ax=ax4)
        ax4.set_title('Loan Status by Property Area')
        buf4 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf4, format='png')
        plt.close(fig4)
        plots['Property Area'] = base64.b64encode(buf4.getvalue()).decode('utf-8')

        # Convert first 10 records to dict for HTML table
        data_table = df.head(10).to_dict(orient='records')

    except Exception as e:
        print(f"EDA error: {e}")

    return render(request, 'pages/eda.html', {'plots': plots, 'data_table': data_table})

# Data Issues View
def data_issues_view(request):
    issue_plot = None
    df_path = os.path.join(settings.BASE_DIR, 'loan_prediction', 'dataset', 'loan_prediction.csv')

    try:
        df = pd.read_csv(df_path)
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            fig, ax = plt.subplots()
            missing.plot(kind='barh', ax=ax)
            ax.set_title('Missing Values in Dataset')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            issue_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Data issue plot error: {e}")

    return render(request, 'pages/data_issues.html', {'issue_plot': issue_plot})

# Model Evaluation View
def evaluation_view(request):
    metrics_path = os.path.join(settings.BASE_DIR, 'loan_prediction', 'model', 'lr_metrics.json')
    metrics = {}

    try:
        with open(metrics_path, 'r') as f:
            raw_metrics = json.load(f)

            metrics['accuracy'] = raw_metrics.get('accuracy', 'N/A')
            metrics['confusion_matrix'] = raw_metrics.get('confusion_matrix', [])

            classification_report = raw_metrics.get('classification_report', {})
            safe_report = {}
            for label, stats in classification_report.items():
                if isinstance(stats, dict):
                    cleaned_stats = {
                        k.replace('-', '_'): v for k, v in stats.items()
                    }
                    safe_report[label] = cleaned_stats
                else:
                    safe_report[label] = stats

            metrics['classification_report'] = safe_report

    except FileNotFoundError:
        metrics['error'] = "Failed to load evaluation metrics."

    return render(request, 'pages/evaluation.html', {'metrics': metrics})
