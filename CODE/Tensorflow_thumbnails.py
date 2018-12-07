# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:00:16 2018

@author: manor
"""

#Libraries
import re
from datetime import datetime, date,timedelta, time as timepart 
import os.path
import random
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as BeautifulSoup
import sys
import pandas as pd
import numpy as np

#Paths
MainPath=r'C:\Users\manor\Desktop\Mainproject'
DataPath=MainPath+'\\Output\\'
AudioPath=MainPath+'\\Trending\\'
PicturePath=MainPath+'\\Images\\'
#Read the File YT_Top200TagsExtra.xlsx

p1=pd.ExcelFile(DataPath+'YT_Top200TagsExtra.xlsx')
ThumbnailsDF=p1.parse("Sheet1")
ThumbnailsDF.head()

#fix the tags column
ThumbnailsDF['tags']=ThumbnailsDF['tags'].str.replace('[','').str.replace(']','').str.replace('\"','').str.replace("'",'').str.replace("nan",'')


#Download all iamges in a directory and name them image0.p onwards.
#Create labels using the entity recognition from

import nltk
from nltk.corpus import stopwords

#Read custom stop words
custom_stop_words = []
with open( DataPath+'customstopwords.txt', "r" ) as fin:
    for line in fin.readlines():
        custom_stop_words.append( line.strip() )
        
# note that we need to make it hashable
print("Stopword list has %d entries" % len(custom_stop_words) )

custom_stop_words=custom_stop_words+stopwords.words('english')

#Define function for text tagger
def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in custom_stop_words])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
                elif chunk.label() == 'LOCATION' or chunk.label() == 'FACILITY' or chunk.label() == 'GPE' or chunk.label()=='ORGANIZATION':
                    names.append(' '.join([c[0] for c in chunk]))
                else:
                    names.append(' '.join([c[0] for c in chunk]))
    return names

from string import ascii_letters, punctuation
allowed = set(ascii_letters)
#allnamescleanlist = [word for word in allnameslistnew if any(letter in allowed for letter in word)]

allnameslist=[]
for index, row in ThumbnailsDF.iterrows():
    print(index)
    listrow=str(row['tags']).split(",")
    outalllist=extract_names(row['video_title'])
    if str(listrow[:1]).lower()!='nan':
        allnameslist.append(str(listrow[:1]).lstrip())
    elif [word for word in outalllist[:1] if any(letter in allowed for letter in word)]:
        allnameslist.append(str(outalllist[:1]))
    else:
        allnameslist.append(str(outalllist[2]))
        
allnameslistnew = [re.sub("[:\-()\[\]\_\'\.\# ]"," ",x) for x in allnameslist]  
        
allnamescleanlist = [re.sub(' +',' ',x).lstrip().rstrip().lower() for x in allnameslistnew]

len(allnamescleanlist)
ThumbnailsDF['Image_Path']=allnamescleanlist

#import requests
#def downloadImage(pic_url, outfilename):
#    with open(outfilename, 'wb') as handle:
#        response = requests.get(pic_url, stream=True)
#        if not response.ok:
#            print(response)
#        for block in response.iter_content(1024):
#            if not block:
#                break
#            handle.write(block)
#
#ImageChecklist=[]
#counter=1
#for index, row in ThumbnailsDF.iterrows():
#    print(row['thumbnail_link'])
#    FilePath=PicturePath+str(row['Image_Path'])+'.'+str(counter)+'.jpg'
#    print(FilePath)
#    try:
#        downloadImage(row['thumbnail_link'],FilePath)
#        Flag='YES'
#        ImageChecklist.append(Flag)
#    except:
#        Flag='NO'
#        ImageChecklist.append(Flag)
#    counter=counter+1
#  
#TensorFlow Model    
    
import cv2    
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import sys
import matplotlib.pyplot as plt


value = np.empty((), dtype=object)
value=0
empty256 = np.full((256, 256), value, dtype=np.uint8).ravel()
type(empty256)
ImageDataList=[]
counter=1
idxdeletelist=[]
for index, row in ThumbnailsDF.iterrows():
    FilePath=PicturePath+str(row['Image_Path'])+'.'+str(counter)+'.jpg'
    print(FilePath)
    img=cv2.imread(FilePath,cv2.IMREAD_GRAYSCALE)
    try:
        img=cv2.resize(img, (256,256))
        ImageDataList.append(np.array(img))
    except:
        ImageDataList.append(empty256)
        print(index)
        idxdeletelist.append(index)
    counter=counter+1
    
ThumbnailsDF['Image_Data']=ImageDataList

ThumbnailsfinalDF=ThumbnailsDF.drop(ThumbnailsDF.index[idxdeletelist])
ThumbnailsfinalDF.shape

#get count of distinct labels
ThumbnailsfinalDF['Image_Path'].nunique()
        
from keras.models import Sequential
from keras.layers import InputLayer,Conv2D,MaxPool2D,MaxPooling2D,Dropout,Flatten, Dense
from keras.optimizers import Adam

seed = 23; 
np.random.seed(seed); 
tf.set_random_seed(seed) 
tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.initialize_all_variables())

learning_rate = 0.01
epochs_value = 5
batch_size_value = 128

def plot_images(images, cls_true, cls_pred=None):
    fig, axes = plt.subplots(1, 5)
    fig.subplots_adjust(hspace=1, wspace=0.2)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(256,256), cmap='binary')
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "Actual: {0}".format(cls_true[i])
        else:
            xlabel = "Actual: {0},\n Prediction: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)       
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])  
    plt.show()

from sklearn.model_selection import train_test_split
training_set ,test_set = train_test_split(ThumbnailsfinalDF,test_size=0.25)

training_images = training_set[['Image_Data', 'Image_Path' ]].values.tolist()
testing_images = test_set[['Image_Data', 'Image_Path' ]].values.tolist()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,256,256,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,256,256,1)
tst_lbl_data = np.array([i[1] for i in testing_images])  

print("Initial Training Dataset: ", tr_lbl_data.shape, tr_img_data.shape)
print("Initial Testing Dataset: ", tst_lbl_data.shape, tst_img_data.shape)

plot_images(tr_img_data[0:5], tr_lbl_data[0:5], cls_pred=None)

model = Sequential()
model.add(InputLayer(input_shape=[256,256,1]))
model.add(Conv2D(filters=1024,kernel_size=3,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=256,kernel_size=3,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

#model.add(tf.keras.layers.Dense(10,  input_shape=([256,256,1],), activation="sigmoid", kernel_initializer = tf.random_normal_initializer))
#model.add(tf.keras.layers.Dense(1,kernel_initializer = tf.random_normal_initializer)) 
#model.summary()
#layer_input = model.layers[0]
#print(layer_input.get_weights()[0].shape)

#model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))

model.add(Dense(490, activation='softmax'))
model.summary()

#layer_input = model.layers[6]
#print(layer_input.get_weights()[0].shape)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

print (tr_img_data.shape)

training_model = model.fit(tr_img_data, tr_lbl_data, 
                           batch_size=batch_size_value, 
                           epochs=epochs_value,
                           validation_data=(tst_img_data, tst_lbl_data))

model.summary()

training_model.history['acc'], training_model.history['val_acc']

model.save('model.keras', overwrite=True)

result = model.evaluate(tst_img_data, tst_lbl_data)

for name, value in zip(model.metrics_names, result):
    print (name, value)

