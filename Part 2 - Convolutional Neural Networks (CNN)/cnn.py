# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:40:21 2022

@author: Abigail
"""

#Redes neuronales convolucionales

#######################Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
#pip3 install Theano para Windows si no está instalado git 

#######################Instalar Tensorflow y Keras desde anaconda prompt
# conda install -c conda-forge keras
#pip3 install tensorflow

#En la carpeta de dataset ya viene divididas las imágenes 80/20. 
#Hay 8000 imagenes de entrenamiento (mita de perros, mitad de gatos)
# las otras 2000 son el conjunto de test (mitad de perros, mitad de gatos)



###################### Parte 1, construir el modelo de red neuronal convolucional
#importar la librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Inicializar la CNN
classifier = Sequential()


##Paso 1, capa de vonvolución, tomar la imagen y aplicar los filtros para obtener 
#el mapa de características (imagen de entrada*detector de rasgos=mapa de caracteristicas)
#el resultado de este paso es un conjunto de mapas de caracteristicas que forman
#la capa de convolucion
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))
#red con 32 mapas de caracteristicas de 3x3
#input_shape para indicar el tamaño y color de las imagenes(filas, columnas, canales de color),
# intentar tener imágenes de tamaño pequeño para evitar el coste computacional



###Paso 2, MaxPooling
#Del paso anterior tomar una ventana con el máximo de los numeros comprendidos 
#en el interior de esa ventana
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

##Paso 3 Flattening (aplanado de datos)
#pasar del conjunto de capas de pooling a un solo vector (una dimensión vertical)
#que es lo que se necesita para hacer la capa de entrada de la red
classifier.add(Flatten())


##Paso 4 full connection (red totalmente conectada)
classifier.add(Dense(units= 128, activation="relu"))
classifier.add(Dense(units= 1, activation="sigmoid"))


###Compilar la red neuronal de convolución
classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=["accuracy"])


######################### Parte 2 Ajustar la red neuronal  a las imágenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

batch_size = 32
classifier.fit(training_dataset,
               steps_per_epoch=int(8000/32),
               epochs=25,
               validation_data=testing_dataset,
               validation_steps=int(2000/32))



########### Hacer una nueva predicción
import numpy as np
import tensorflow as tf

test_image = tf.keras.preprocessing.image.load_img('dataset/single_prediction/cat_or_dog_5.jpg', target_size = (64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_dataset.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'



