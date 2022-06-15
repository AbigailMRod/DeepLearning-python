# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:36:40 2022

@author: Abigail
"""

#Redes Neuronales Artificiales

#######################Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
#pip3 install Theano para Windows si no está instalado git 


#######################Instalar Tensorflow y Keras desde anaconda prompt
# conda install -c conda-forge keras
#pip3 install tensorflow

#////////////////////////////Parte 1, pre procesado de datos 
#Cómo importar las librerías
import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

###########################################################################
#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#trasnformar los paises en categorías
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])

#transformar el género en categoría
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough')

X = onehotencoder.fit_transform(X)
#para evitar multicolinealidad, se elimina una de las 3 columnas que se generaron 
# en las varibles dummy
X = X[:, 1:]

#########################################################################

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


##El escalado es obligatorio en redes neuronales  
#Escalado de variables 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


################################################################
#/////////////////// Parte 2 - Contruir la Red Neuronal Artificial

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Inicializar la RNA
classifier = Sequential()

#Añadir las capas de entrada y primer capa oculta de la red neuronal
#Dense es la conexion entre capas (la sinapsis)
#units es el número de nodos de la capa oculta, es aceptable utilizar la media entre
#los nodos de la capa de entrada y los nodos de la capa de salida
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
classifier.add(Dropout(p = 0.1))

#relu (rectificador lineal unitario)
# input_dim es la dimension de entrada, en este caso 11 columnas 
#kernel_initializer para mantener lo pesos pequeños, cercanos a 0 pero no nulos

############# segunda capa oculta 
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
classifier.add(Dropout(p = 0.1))

################## Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


####################### Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/cm.sum())




################# Predecir una nueva observación
# Utiliza nuestro modelo de RNA para predecir si el cliente con la siguiente información abandonará el banco:
# Geografia: Francia [0,0] por las variables dummy
# Puntaje de crédito: 600
# Género masculino
# Edad: 40 años de edad
# Tenencia: 3 años.
# Saldo: $ 60000
# Número de productos: 2
# ¿Este cliente tiene una tarjeta de crédito? Sí
# ¿Es este cliente un miembro activo? Sí
# Salario estimado: $ 50000
# Entonces, ¿deberíamos decir adiós a ese cliente?

new_prediction = classifier.predict(sc_X.transform(np.array([[0,0,600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(new_prediction)
print(new_prediction > 0.5)

#resultado : FALSE, es decir, el clinete no abandona el banco


############################### Parte 4 - Evaluar, mejorar y Ajustar la RNA

### Evaluar la **RNA**
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
  classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1, verbose = 1)

mean = accuracies.mean()
variance = accuracies.std()

### Mejorar la RNA
#### Regularización de Dropout para evitar el *overfitting*

### Ajustar la *RNA*
from sklearn.model_selection import GridSearchCV # sklearn.grid_search

def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))
  classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {
    'batch_size' : [25,32],
    'nb_epoch' : [100, 500], 
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




