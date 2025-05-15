from django.urls import path
from . import views

app_name = 'loan_prediction'

urlpatterns = [
    path('home/', views.home, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('requests/', views.manage_requests_view, name='requests'),
    path('eda/', views.eda_view, name='eda'),
    path('data_issues/', views.data_issues_view, name='data_issues'),
    path('evaluation/', views.evaluation_view, name='evaluation'),
]

