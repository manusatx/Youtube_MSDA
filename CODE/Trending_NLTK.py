# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:31:21 2018

@author: manor
"""

#Read the Trending All file
import pandas as pd
import numpy as np

MainPath=r'C:\Users\manor\Desktop\Mainproject'
DataPath=MainPath+'\\Output\\'
AudioPath=MainPath+'\\Trending\\'

p1=pd.ExcelFile(DataPath+'YT_Trending_All.xlsx')
trendingDF=p1.parse("Sheet1")
trendingDF.head()

trendingDF=trendingDF[:84]

#Read each audio text file and add as a column to dataframe
audiotextlist=[]
for x in range(1, len(trendingDF)+1):
    #print(x)
    FilePath=AudioPath+'Trending'+str(x)+'.txt'
    #print(FilePath)
    with open(FilePath, 'r') as myfile:
        data = myfile.read()
    audiotextlist.append(data)
    
trendingDF['audiotext']=audiotextlist

#nltk libraries
#Pos tagger chunk labelling
import nltk
from nltk.corpus import stopwords

#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

#Define function for text tagger
def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stopwords.words('english')])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    places = []
    datetimes=[]
    otherText = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'LOCATION' or chunk.label() == 'FACILITY' or chunk.label() == 'GPE' or chunk.label()=='ORGANIZATION':
                    places.append(' '.join([c[0] for c in chunk]))
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'DATE' or chunk.label() == 'TIME':
                    datetimes.append(' '.join([c[0] for c in chunk]))
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() != 'PERSON':
                    otherText.append(' '.join([c[0] for c in chunk]))
    return names, places, datetimes, otherText

#Find labelled names, places, dates etc
allnameslist=[]
allplaceslist=[]
alldatetimeslist=[]
allotherslist=[]
for index, row in trendingDF.iterrows():
    print(index)
    listrow=str(row['video_title']).split(" ")
    #print(listrow)
    outalllist=extract_names(row['video_title'])
    allnameslist.append(outalllist[0])
    allplaceslist.append(outalllist[1])
    alldatetimeslist.append(outalllist[2])
    allotherslist.append(outalllist[3])

#Find labelled names, places, dates etc
audiotextnameslist=[]
audiotextplaceslist=[]
audiotextdatetimeslist=[]
audiotextotherslist=[]
for index, row in trendingDF.iterrows():
    print(index)
    listrow=str(row['audiotext']).split(" ")
    #print(listrow)
    outalllist=extract_names(row['audiotext'])
    audiotextnameslist.append(outalllist[0])
    audiotextplaceslist.append(outalllist[1])
    audiotextdatetimeslist.append(outalllist[2])
    audiotextotherslist.append(outalllist[3])
    
    
#Get unigrams and bigrams from the video title
from sklearn.feature_extraction.text import CountVectorizer
for index, row in trendingDF.iterrows():
    print(index)
    listrow=str(row['video_title']).split(" ")
    vector_u = CountVectorizer(ngram_range = (1,1), analyzer = 'word', stop_words=stopwords.words('english'))
    unigrams = vector_u.fit(listrow).get_feature_names()
    print(unigrams)
    vector_b = CountVectorizer(ngram_range = (2,2), analyzer = 'word', stop_words=stopwords.words('english'))
    bigrams = vector_b.fit(list(str(row['video_title']).replace('|',',').split(","))).get_feature_names()
    print(bigrams)


#TFIDF and count vectorizer on audiotext
documents=list(trendingDF['audiotext'])
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
no_features = 1000
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stopwords.words('english'))
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stopwords.words('english'))
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

#TOPIC Modelling
from sklearn.decomposition import NMF, LatentDirichletAllocation
topics_count = 20
# Run NMF
nmf = NMF(n_components=topics_count, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# Run LDA
lda = LatentDirichletAllocation(n_components=topics_count, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 4
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)