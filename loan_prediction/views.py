import os
import json
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import LoanRequestForm
from .models import LoanRequest
import base64

# Load logistic regression parameters and scaler stats from JSON once
model_params_path = os.path.join(settings.BASE_DIR, 'loan_prediction', 'model', 'model_params.json')
with open(model_params_path, "r") as f:
    model_params = json.load(f)
    
coefficients = np.array(model_params["coefficients"])
intercept = model_params["intercept"]
mean = np.array(model_params["scaler_mean"])
scale = np.array(model_params["scaler_scale"])

def standard_scale(X):
    """
    Apply standard scaling: (X - mean) / scale
    Input: X - numpy array of raw feature values
    Output: scaled features as numpy array
    """
    return (X - mean) / scale

def logistic_sigmoid(x):
    """
    Compute the logistic sigmoid function element-wise.
    """
    x = np.asarray(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))

def predict_proba(X_raw):
    """
    Predict probability for positive class given raw feature vector.
    Steps:
    - Standard scale features
    - Calculate linear combination with coefficients + intercept
    - Apply sigmoid to get probability
    """
    X_scaled = standard_scale(X_raw)
    linear_output = np.dot(X_scaled, coefficients) + intercept
    prob = logistic_sigmoid(linear_output)
    return prob[0]

def home(request):
    """Render home page."""
    return render(request, 'pages/home.html')

def predict_view(request):
    """
    Handle GET/POST requests for loan prediction.
    - On GET: Show empty form.
    - On POST: Validate form, transform inputs, predict probability using manual logistic regression.
    - Classify into 'Approved', 'Under Review', or 'Rejected' based on probability thresholds.
    - Save form data with predicted loan_status to DB.
    """
    result = None
    probability = None

    if request.method == 'POST':
        form = LoanRequestForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Construct feature vector in exact order used in model training:
            # Note: No log transform is applied here because training was done without it.
            # Convert categorical string values to binary indicators as model expects.
            input_features = np.array([
                data['applicant_income'],                  # ApplicantIncome
                data['coapplicant_income'],                # CoapplicantIncome
                data['loan_amount'],                       # LoanAmount
                float(data['loan_amount_term']),           # Loan_Amount_Term 
                float(data['credit_history']),             # Credit_History
                int(data['gender'] == 'Male'),             # Gender_Male
                int(data['married'] == 'Yes'),             # Married_Yes
                int(data['dependents'] == '1'),            # Dependents_1
                int(data['dependents'] == '2'),            # Dependents_2
                int(data['dependents'] in ['3', '3+']),    # Dependents_3
                int(data['education'] != 'Graduate'),      # Education_Not Graduate 
                int(data['self_employed'] == 'Yes'),       # Self_Employed_Yes
                int(data['property_area'] == 'Semiurban'), # Property_Area_Semiurban
                int(data['property_area'] == 'Urban')      # Property_Area_Urban
            ]).reshape(1, -1)

            # Calculate probability using manual logistic regression prediction
            prob = predict_proba(input_features)
            probability = round(prob * 100, 2)  # percentage

            # Classification thresholds for loan decision
            if prob >= 0.6:
                result = "Approved"
            elif prob >= 0.5:
                result = "Under Review"
            else:
                result = "Rejected"

            # Save loan request with predicted loan_status to database
            loan_request = form.save(commit=False)
            loan_request.loan_status = result
            loan_request.save()
    else:
        form = LoanRequestForm()

    return render(request, 'pages/predict.html', {
        'form': form,
        'result': result,
        'probability': probability
    })

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
        queryset = LoanRequest.objects.all()
        df = pd.DataFrame(list(queryset.values()))

        if df.empty:
            raise ValueError("No loan requests in the database.")

        # Map DB fields to dataframe columns expected by plots
        df['Credit_History'] = df['credit_history'].fillna(0)
        df['LoanAmount'] = df['loan_amount']
        df['ApplicantIncome'] = df['applicant_income']
        df['Loan_Status'] = df['loan_status']
        df['Property_Area'] = df['property_area']

        # Scatter plot: Applicant Income vs Loan Amount
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

        # Boxplot: Loan Amount by Loan Status
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Loan_Status', y='LoanAmount', data=df, ax=ax2)
        ax2.set_title('Loan Amount Distribution by Status')
        buf2 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png')
        plt.close(fig2)
        plots['Loan Amount by Status'] = base64.b64encode(buf2.getvalue()).decode('utf-8')

        # Countplot: Credit History vs Loan Status
        fig3, ax3 = plt.subplots()
        sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=ax3)
        ax3.set_title('Loan Status by Credit History')
        buf3 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf3, format='png')
        plt.close(fig3)
        plots['Credit History'] = base64.b64encode(buf3.getvalue()).decode('utf-8')

        # Countplot: Property Area vs Loan Status
        fig4, ax4 = plt.subplots()
        sns.countplot(x='Property_Area', hue='Loan_Status', data=df, ax=ax4)
        ax4.set_title('Loan Status by Property Area')
        buf4 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf4, format='png')
        plt.close(fig4)
        plots['Property Area'] = base64.b64encode(buf4.getvalue()).decode('utf-8')

        # Prepare first 10 records for table display
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
