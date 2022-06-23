# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:07:23 2022

@author: Abigail
"""

#Redes neuronales recurrentes 

#................ Parte1 Preprocesado de datos
#Importación de las librerias

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Importar el dataset de entrenamiento
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set  = dataset_train.iloc[:, 1:2].values

# Escalado de las características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Crear una estructura de datos con 60 timesteps y 1 salida
#60 días qe tratará de entender para predecir a partir de ello
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Redimension de los datos 
#Reshape con fila, columna, profundidad
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#................ Parte 2 Construcción de la RNR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Inicialización del modelo
regressor = Sequential()

#Añadir la primera capa de LSTM y la regularización por Dropout para evitar el sobreajuste
#numero de celdas, de unidades que queremos 
#secuancia de retorno (True, si es un LSTM apilado)
#tamño de la forma de entrada que deben tener los datos 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
regressor.add(Dropout(0.2))

#Añadir la segunda capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

#Añadir la tercera capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

#Añadir la cuarta capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


#Anadir la capa de salida
regressor.add(Dense(units = 1))


#Compilar la red neuronal recurrente
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


#Ajustar la red al conjunto de datos de entrenamiento
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#............... Parte 3 Ajustar las predicciones y visualizar los resultados
#Obtener el valor real de las aaciones de enero de 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price  = dataset_test.iloc[:, 1:2].values

#Predecir las acciones de enero de 2017 con la red recurrente

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1) 
inputs = sc.transform(inputs)

X_test =[]
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualización de los resultados
plt.plot(real_stock_price, color='blue', 
         label= 'Precio real de la acción de Google')
plt.plot(predicted_stock_price, color='green', 
         label= 'Precio predicho de la acción de Google')
plt.title("Predicción de la Red Neuronal Recurrente del valor de las acciones de Google")
plt.xlabel("Fecha")
plt.ylabel("Precio de la acción de Google")
plt.legend()






















