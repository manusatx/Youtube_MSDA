# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:16:03 2018

@author: Ghi
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn 
import gc; gc.enable()


learning_rate = 0.01
epochs_value = 100
batch_size_value = 8

FEATURES = ["TAG1","TAG2","TAG3","TAG4","TAG5","TAG6","TAG7","TAG8","TAG9","TAG10"]

LABEL3 ="like_count"	
LABEL4 ="Above_med_like"	
LABEL2 ="view_count"	
LABEL1 ="Above_med_view"
LABEL5 ="Above_75Perctntile_view"
LABEL6 ="Above_80Perctntile_view"
seed = 31912; 
np.random.seed(seed); 
tf.set_random_seed(seed) 

#load dataset
path= "C:/Users/Ghi/Documents/MSDA/Big Data Technology/PROJECT/YT_Top200TagsBcounts.csv"
data_ =pd.read_csv(path)
data=pd.DataFrame(data_)

#Training and Test Split 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.25)
#Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,  input_shape=(len(FEATURES),), activation="sigmoid", kernel_initializer = tf.random_normal_initializer))
model.add(tf.keras.layers.Dense(100, activation="softmax", kernel_initializer = tf.random_normal_initializer))
model.add(tf.keras.layers.Dense(25, activation="sigmoid", kernel_initializer = tf.random_normal_initializer))
model.add(tf.keras.layers.Dense(1,activation="sigmoid", kernel_initializer = tf.random_normal_initializer)) 


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#Train the model
training_model = model.fit(training_set[FEATURES].values, training_set[LABEL1].values,epochs=epochs_value,batch_size=batch_size_value,validation_data=(test_set[FEATURES].values, test_set[LABEL1].values))

#Making Predictions
y_pred = model.predict_proba(x=test_set[FEATURES].values) #, verbose=1)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

confusion_matrix(test_set[LABEL1].values, y_pred)
accuracy_score(test_set[LABEL1].values, y_pred)

print(sklearn.metrics.classification_report(test_set[LABEL1].values, y_pred))



