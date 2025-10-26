import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle

import streamlit as st


#load the saved model, scaler and encoders
model = load_model('churn_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

#stremalit app for churn prediction
st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn probability.")
#input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", one_hot_encoder.categories_[0].tolist())
gender   = st.selectbox("Gender", label_encoder.classes_.tolist())
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

#prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': label_encoder.transform([gender]),
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

##one hot encode geography
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography'])) 

#combine the encoded geography columns with the input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)


#scale the features
input_data_scaled = scaler.transform(input_data)

#predict churn
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    if churn_probability > 0.5:
        st.error(f"Customer is likely to churn with probability {churn_probability:.2f}")
    else:
        st.success(f"Customer is unlikely to churn with probability {churn_probability:.2f}")
    


