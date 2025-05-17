from django.db import models

# Model to store loan request information and prediction results
class LoanRequest(models.Model):
    gender_choices = [('Male', 'Male'), ('Female', 'Female')]
    married_choices = [('Yes', 'Yes'), ('No', 'No')]
    education_choices = [('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')]
    self_employed_choices = [('Yes', 'Yes'), ('No', 'No')]
    property_area_choices = [('Urban', 'Urban'), ('Rural', 'Rural'), ('Semiurban', 'Semiurban')]
    dependents_choices = [('0', '0'), ('1', '1'), ('2', '2'), ('3', '3')]
    credit_history_choices = [(0.0, '0'), (1.0, '1')]

    gender = models.CharField(max_length=10, choices=gender_choices)
    married = models.CharField(max_length=10, choices=married_choices)
    dependents = models.CharField(max_length=5, choices=dependents_choices)
    education = models.CharField(max_length=20, choices=education_choices)
    self_employed = models.CharField(max_length=10, choices=self_employed_choices)
    applicant_income = models.FloatField(default=0) 
    coapplicant_income = models.FloatField(default=0)
    loan_amount = models.FloatField()
    loan_amount_term = models.FloatField()
    credit_history = models.FloatField(choices=credit_history_choices)
    property_area = models.CharField(max_length=20, choices=property_area_choices)
    loan_status = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f"Request by {self.gender}, Income: {self.applicant_income}, Status: {self.loan_status}"
