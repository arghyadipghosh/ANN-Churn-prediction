import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model

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

def predict_input_data(df):

  # sc = StandardScaler()
  with open(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\scaler.pkl','rb') as file:
    scaler = pickle.load(file)
  with open(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\le_gender.pkl','rb') as file:
    le_gender = pickle.load(file)
  with open(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)

  # df = pd.DataFrame(input_data, index=[1])
  print(df)

  # le = LabelEncoder()
  df['Gender'] = le_gender.fit_transform(df['Gender'])
  df = df.reset_index(drop=True)
  print(f'After encoding gender columns df')
  print(df)

  # ohe = OneHotEncoder()
  geo_ohe = ohe_geo.transform(df[['Geography']]).toarray()
  print(ohe_geo.get_feature_names_out())

  enc_df = pd.DataFrame(geo_ohe,columns=ohe_geo.get_feature_names_out(['Geography']))
  print('Encoded df:')
  enc_df  =enc_df.reset_index(drop=True)
  print(enc_df)

  df = pd.concat([(df.drop('Geography',axis=1)),enc_df],axis=1)
  print('Final df:')
  print(df)



  print(f'scaler: {scaler}')
  df_scaled = scaler.transform(df)
  print(df_scaled)

  model = load_model(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\model.h5')

  prediction = model.predict(df_scaled)
  prediction_pba = prediction[0][0]
  print(prediction_pba)
  status=''
  if prediction_pba > 0.5:
    status = 'The customer is likely to churn'
    print('The customer is likely to churn')
  else:
    status = 'The customer is not likely to churn'
    print('The customer is not likely to churn')

  return prediction_pba, status






