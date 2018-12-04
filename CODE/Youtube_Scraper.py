a# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Selenium scraper
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
from datetime import datetime, date,timedelta, time as timepart 
import os.path
import random
from bs4 import BeautifulSoup as soup
import sys

##Start time of script
nowDTG = datetime.now().replace(microsecond=0)
print('Script started at: ' + str(nowDTG))

#Writw data to file
def write_data_to_file(fname,data):
    f=open(fname,'w')
    f.write(str(data))
    f.close()
    print("File Write done!")
    return

#Make alist of known folks whose audio we want to download
Persons = ['Ray Dalio interview', 'Kenneth Arrow' , 'James P. Gorman interview', 'Jamie Dimon interview' ]
    
### Driver
webDriverLocation = r'C:\Users\manor\Desktop\Chromedriver\chromedriver.exe'

seleniumHeadless = False
browser = 'chrome'

if seleniumHeadless:
    if (browser == 'chrome'):
        from selenium.webdriver.chrome.options import Options
        #options = webdriver.ChromeOptions()
        options = Options()
        options.add_argument('--headless')
        options.add_argument("--disable-notifications")
        options.add_argument("--dns-prefetch-disable")
        driver = webdriver.Chrome(executable_path=webDriverLocation, options=options) #hidden browser (lighter on resources)
    elif (browser == 'firefox'):
        from selenium.webdriver.firefox.options import Options
        options = Options()
        options.add_argument('--headless')
        driver = webdriver.Firefox(firefox_options=options, executable_path=webDriverLocation)
    else:
        print('You need to specify a webdriver to use')
        sys.exit()
else:
    if (browser == 'chrome'):
        from selenium.webdriver.chrome.options import Options                                                                                                                         
        options = Options()
        options.add_argument("--disable-notifications")
        driver = webdriver.Chrome(executable_path=webDriverLocation, options=options) #unhidden browser (so you can watch what is happening)
    elif (browser == 'firefox'):
        driver = webdriver.Firefox(executable_path=webDriverLocation)
    else:
        print('You need to specify a webdriver to use')
        sys.exit()
        
#Connect to YOutube and click on Search Button
#baseUrl=('https://www.youtube.com/')
#driver.get(baseUrl)
#driver.find_element_by_id('search')

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import youtube_dl
AudioPath=r'C:\Users\manor\Desktop\Mainproject'
ffmpegexec=r'C:\Users\manor\Desktop\ffmpeg\bin\ffmpeg.exe'

import speech_recognition as sr
#import sphinxbase
#import pocketsphinx

import os
import subprocess

for per in Persons:
    baseUrl=('https://www.youtube.com/')
    driver.get(baseUrl)
    wait = WebDriverWait(driver, 10)
    search_input = wait.until(EC.presence_of_element_located((By.ID, "search")))
    search_input.send_keys(per)
    search_input.send_keys(u'\ue007')
    wait = WebDriverWait(driver, 10)   
    elems = driver.find_element_by_id("content").find_elements_by_id("video-title")
    youtubealllinks= []
    for elem in elems:
        #print(elem)
        link=elem.get_attribute("href")
        #print(link)
        time.sleep((random.randrange(5, 9, 1))*0.1)
        youtubealllinks.append(link)
        #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        #time.sleep((random.randrange(2, 5, 1))*0.1)
    #get top 5 links only from each list
    youtubelinks = youtubealllinks[:5]
    #print(youtubelinks)
    counter=1
    for link in youtubelinks:
        print(link)
        #testmp3= AudioPath+'/UBkHdwF8yv0'
        testmp3=AudioPath+'\\'+per+str(counter)+'.mp3'
        testwav=AudioPath+'\\'+per+str(counter)+'.wav'
        testtext=AudioPath+'\\'+per+str(counter)+'.txt'
        #options = {'outtmpl': '%(id)s'}  # save file as the YouTube ID
        options = {
          'format': 'bestaudio/best',
          'extractaudio' : True,  # only keep the audio
          'audioformat' : "mp3",  # convert to mp3
          #'outtmpl': '%(id)s',    # name the file the ID of the video
          'noplaylist' : True,    # only download single song, not playlist
          'outtmpl': testmp3
        } 
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([link])
        #Remove old file if its there
        try:
            os.remove(testwav)
        except:
            print("No File, its ok")
        #call ffmpeg to convert to wave
        seconds = "4"
        subprocess.call([ffmpegexec, '-i', testmp3, '-ss', seconds, testwav])

        #Convert wav to txt using pocketsphinx
        sr.__version__
        r=sr.Recognizer()
        testaudio=sr.AudioFile(testwav)
        with testaudio as source:
            audio = r.record(source)
        type(audio)
        textdata = r.recognize_sphinx(audio)
        write_data_to_file(testtext,textdata)
        counter = counter + 1   
        
##End time of script
nowDTG = datetime.now().replace(microsecond=0)
print('Script ended at: ' + str(nowDTG))


#Fix the data in text using autocorrect
from autocorrect import spell
import pandas as pd

AudioPath=r'C:\Users\manor\Desktop\Mainproject\\'
filedf=pd.DataFrame()
filedf['Text'] = pd.read_csv(AudioPath+'James P. Gorman interview2.txt', header=None)
filedf['Name'] = 'James P. Gorman interview'
filedf['ID'] = 1

NewCorrText = []
for i in filedf[['Text']].to_string(index=False).split(' '):
    correctedtext = spell(i)
    print(i+' '+correctedtext)
    NewCorrText.append(correctedtext)

NewCorrTextMerged = ' '.join(NewCorrText)

filedf['NewText'] = NewCorrTextMerged


for i in filedf['Text'].to_string(index=False):
    print(i)
    
    
#wordtovec and tesnorflow training to fix language
#import nltk
#nltk.download('stopwords')

#type(filedf['Text'])
#see column names and data types of columns in panda df
filedf.apply(lambda x: pd.api.types.infer_dtype(x.values))

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

vector_u = CountVectorizer(ngram_range = (1,1), analyzer = 'word', stop_words=stopwords.words('english'))
unigrams = vector_u.fit(filedf['NewText'] ).get_feature_names()

vector_b = CountVectorizer(ngram_range = (2,2), analyzer = 'word', stop_words=stopwords.words('english'))
bigrams = vector_b.fit(filedf['NewText'] ).get_feature_names()

vector_t = CountVectorizer(ngram_range = (3,3), analyzer = 'word', stop_words=stopwords.words('english'))
trigrams = vector_t.fit(filedf['NewText'] ).get_feature_names()

vector_q = CountVectorizer(ngram_range = (4,4), analyzer = 'word', stop_words=stopwords.words('english'))
quadgrams = vector_q.fit(filedf['NewText'] ).get_feature_names()

vector_p = CountVectorizer(ngram_range = (5,5), analyzer = 'word', stop_words=stopwords.words('english'))
pentagrams = vector_p.fit(filedf['NewText'] ).get_feature_names()

#Pos tagger chunk labelling
import nltk
stopw = vector_u.get_stop_words()

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stopwords.words('english')])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    otherText = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() != 'PERSON':
                    otherText.append(' '.join([c[0] for c in chunk]))
    return names, otherText

ie_preprocess(filedf['Text'])


#Get labelled data of these bigrams


#Topic Modelling
#Latent Dirichlet allocation (LDA) 
ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))
ap_lda

