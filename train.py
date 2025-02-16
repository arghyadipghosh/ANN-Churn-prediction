import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras import Sequential


# pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect width (or set a specific value)
# pd.set_option('display.max_colwidth', None) # Show full content of each column

def get_data():
  df = pd.read_csv(r"C:\Users\acer\python files\Churn_Modelling.csv") 
  print(f"Fetched csv data: {df.columns}")

  df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'],axis=1)
  print(f"After dropping columns df")
  print(df)

  le = LabelEncoder()
  df['Gender'] = le.fit_transform(df['Gender'])
  print(f'After encoding gender columns df')
  print(df)
  with open(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\le_gender.pkl','wb') as file:
    pickle.dump(le,file)

  ohe = OneHotEncoder()
  geo_ohe = ohe.fit_transform(df[['Geography']])
  print(ohe.get_feature_names_out())
  with open(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\ohe_geo.pkl','wb') as file:
    pickle.dump(ohe,file)

  enc_df = pd.DataFrame(geo_ohe.toarray(),columns=ohe.get_feature_names_out(['Geography']))
  print('Encoded df:')
  print(enc_df)

  df = pd.concat([(df.drop('Geography',axis=1)),enc_df],axis=1)
  print('Final df:')
  print(df)
  print(df.isnull().sum())

  x_train = df.drop('Exited',axis=1)
  y_train = df['Exited']

  random_state = np.random.randint(0,1000)

  x_train,x_test,y_train, y_test = train_test_split(x_train, y_train,test_size=0.2,random_state=random_state)

  sc=StandardScaler()
  x_train = sc.fit_transform(x_train)
  x_test = sc.transform(x_test)
  with open(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\scaler.pkl','wb') as file:
    pickle.dump(sc,file)

  from tensorflow.keras.layers import Dense
  from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

  model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)), #HL1 connected to the input layer
    Dense(32, activation='relu'), #HL2
    Dense(16, activation='relu'), #HL3
    Dense(1, activation='sigmoid'), #Output layer
  ])

  print(model.summary())

  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  import datetime
  log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d")
  tensorflow_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
  early_stopping_callback = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

  history = model.fit(
    x_train, y_train,
    validation_data =(x_test,y_test),
    epochs=100, 
    callbacks =[tensorflow_callback,early_stopping_callback]

  )
  print('model training done!')
  model.save(r'C:\Users\acer\python files\GenAI\ANN_IMPLMENTATION\model.h5')










  
  

if __name__ == "__main__":
  get_data()
