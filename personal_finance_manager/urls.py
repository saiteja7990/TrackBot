"""personal_finance_manager URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from UserApp import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index),
    path('RegAction',views.RegAction),
    path('LogAction',views.LogAction),
    path('home',views.home),
    path('Upload',views.Upload),
    path('UploadAction',views.UploadAction),
    path('ViewDataset',views.ViewDataset),
    path('monthly',views.monthly),
    path('monthly_income',views.monthly_income),
    path('Cumulative',views.Cumulative),
    path('AddExpenses',views.AddExpenses),
    path('AddEIAction',views.AddEIAction),
    path('Chatbot',views.Chatbot),
    path('ChatAction',views.ChatAction),
    path('expense_chart',views.expense_chart),
    path('Predict_Expenses',views.Predict_Expenses),
]
