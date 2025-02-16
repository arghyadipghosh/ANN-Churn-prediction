import streamlit as st
import numpy as np
import pandas as pd
from prediction import predict_input_data

st.title('Customer Churn Prediction')


# test_input_data={
#     'CreditScore':709,
#     'Gender':'Male',
#     'Age':36,
#     'Tenure':7,
#     'Balance':0,
#     'NumOfProducts':1,
#     'HasCrCard':0,
#     'IsActiveMember':1,
#     'EstimatedSalary':101348.88,
#     'Geography':'Spain'

# }

# Input fields
credit_score = st.number_input("Credit Score", min_value=0, value=600)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, value=2)
balance = st.number_input("Balance", min_value=0, value=0)
num_of_products = st.number_input("Number of Products", min_value=0, value=2)
has_cr_card = st.checkbox("Has Credit Card")
is_active_member = st.checkbox("Is Active Member")
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=60000)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Create DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender], # keep the string value
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [int(has_cr_card)],  # Convert boolean to integer
    'IsActiveMember': [int(is_active_member)], # Convert boolean to integer
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})

if st.button('Enter'):
  print('inside enter if check')
  print(input_data)
  print("*************")
  pred, status = predict_input_data(input_data)

  st.write(f"The prediction score is: {pred}")
  st.write(f"The prediction is : {status}")

