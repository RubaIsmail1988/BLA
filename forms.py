from django import forms
from .models import LoanRequest

class LoanRequestForm(forms.ModelForm):
    class Meta:
        model = LoanRequest
        fields = [
            'gender',
            'married',
            'dependents',
            'education',
            'self_employed',
            'applicant_income',
            'coapplicant_income',
            'loan_amount',
            'loan_amount_term',
            'credit_history',
            'property_area',
        ]
        widgets = {
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'married': forms.Select(attrs={'class': 'form-select'}),
            'dependents': forms.Select(attrs={'class': 'form-select'}),
            'education': forms.Select(attrs={'class': 'form-select'}),
            'self_employed': forms.Select(attrs={'class': 'form-select'}),
            'property_area': forms.Select(attrs={'class': 'form-select'}),
            'applicant_income': forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}),
            'coapplicant_income': forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}),
            'loan_amount': forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}),
            'loan_amount_term': forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}),
            'credit_history': forms.NumberInput(attrs={'class': 'form-control', 'step': 'any'}),
        }
