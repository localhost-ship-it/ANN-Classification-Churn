import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle


#load the saved model, scaler and encoders
model = load_model('churn_model.h5')

#load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

#example new data for prediction
input_data = pd.DataFrame({
    'CreditScore': [600],
    'Geography': ['France'],
    'Gender': ['Male'],
    'Age': [40],    
    'Tenure': [3],
    'Balance': [60000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [50000]
})  

geo_encoded = one_hot_encoder.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_data = input_data.drop(['Geography'], axis=1)
print("Input data after encoding:")
print(input_data)

#encode gender
input_data ['Gender'] = label_encoder.transform(input_data['Gender'])
print("Input data after gender encoding:")
print(input_data)

#scale the features
input_data_scaled = scaler.transform(input_data)
print("Scaled input data:")
print(input_data_scaled)

#predict churn
prediction = model.predict(input_data_scaled)
churn_probability = prediction[0][0]
if churn_probability > 0.5:
    print(f"Customer is likely to churn with probability {churn_probability:.2f}")
else:
    print(f"Customer is unlikely to churn with probability {churn_probability:.2f}")    
