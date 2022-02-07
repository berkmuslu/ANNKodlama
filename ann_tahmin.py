# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 19:30:56 2022

@author: Berk Muslu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#### Preprocessing Part ####

data = pd.read_csv('veri.csv')
data.dropna(inplace=True)

data.reset_index(inplace=True)
data.drop(['Hafta', 'index','Tarih','Eski Tarih','Makine Grubu','Makine'], axis = 1, inplace=True)

import seaborn as sn
sn.heatmap(data.corr())

training_set = data.iloc[:2428,:].values
test_set = data.iloc[2428: ,:].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

test_set_scaled = test_set_scaled[:, 0:5]

X_train = []
y_train = []
WS = 40

for i in range(WS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i, 0:6])
    y_train.append(training_set_scaled[i,5])
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))

#### Creating ANN ####

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

Model = Sequential()

Model.add(LSTM(units = 150, return_sequences= True, input_shape = (X_train.shape[1], 6)))
Model.add(Dropout(0.2))


Model.add(LSTM(units = 150, return_sequences= True))
Model.add(Dropout(0.2))


Model.add(LSTM(units = 150, return_sequences= True))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 150))
Model.add(Dropout(0.2))

Model.add(Dense(units = 1))

Model.compile(optimizer = 'adam', loss = 'mean_squared_error')
Model.fit(X_train, y_train, epochs = 40, batch_size = 32)
Model.save('ANN-Model2')


# To Load the last model #
from keras.models import load_model
Model = load_model('ANN-Model')


##### Checking if epochs is enough or not
plt.plot(range(len(Model.history.history['loss'])),Model.history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#### Prediction Part ####

prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,6))

for i in range(247): #next 1 weeks
    first_prediction = Model.predict(Batch_New)[0]
    prediction_test.append(first_prediction)
    
    New_var = test_set_scaled[i,:]
    
    New_var = New_var.reshape(1,5)
    
    New_test = np.insert(New_var, 2, [first_prediction], axis = 1)
    
    New_test = New_test.reshape(1,1,6)
    
    Batch_New = np.append(Batch_New[:,1:,:],New_test,axis = 1)
    
prediction_test = np.array(prediction_test)

SI = MinMaxScaler(feature_range=(0,1))

y_Scale = training_set[:,5:6]
SI.fit_transform(y_Scale)

predictions = SI.inverse_transform(prediction_test)

#### Checking the results on graph

real_values = test_set[:, 5]

plt.plot(real_values, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'Predicted Values')
plt.title('Net Çevrim Üretim Dakikası Tahmin')
plt.xlabel('Time')
plt.ylabel('Net Çevrim Üretim Dakikası')
plt.legend()
plt.show()

#### Success Percentage

def mean_absolute_percantage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))* 100

print(mean_absolute_percantage_error(real_values, predictions))