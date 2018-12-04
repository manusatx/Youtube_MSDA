# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:14:55 2018

@author: manor
"""

import pandas as pd
import numpy as np

MainPath=r'C:\Users\manor\Desktop\Mainproject'
DataPath=MainPath+'\\Output\\'

#Put this data into Excel
from pandas import ExcelWriter


finaldf=pd.DataFrame()
counter=1
for x in range(1, 4):
    FilePath=DataPath+'YT_Trending'+str(counter)+'.xlsx'
    print(FilePath)
    xl = pd.ExcelFile(FilePath)
    finaldf=finaldf.append(xl.parse("Sheet1"))
    counter=counter+1   
    finaldf=finaldf.reset_index(drop=True)
    
    #Find top ten tags of trending videos
    from collections import Counter
    listtags=list(finaldf['tags'].str.replace('[','').str.replace(']','').str.replace('\"','').str.replace("'",''))
    listtagsstr=','.join(str(e) for e in listtags)
    tagslist=listtagsstr.split(",")
    tagslist = [x.strip('\^ ') for x in tagslist]
  
    #len(tagslist)
    ngrams = [(tagslist[i]) for i in range(len(tagslist)-1)]
    counters_dict=Counter(ngrams)
    countersdf = pd.DataFrame.from_dict(counters_dict, orient='index').reset_index()
    countersdf.columns=['ngram', 'count']
    countersdf=countersdf.sort_values(by=['count'], ascending=False)
    TopTenTags=countersdf.head(10)
    Top20Tags=countersdf.head(20)
    Top200Tags=countersdf.head(200)
    #matplotlib wordcloud lets get top 50 tags  
    Top50Tags=countersdf.head(50)
    
    dcount = {}
    for a, x in Top50Tags.values:
        dcount[a] = x
    
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=dcount)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    

MainPath=r'C:\Users\manor\Desktop\Mainproject'
AudioPath=MainPath+'\\Trending\\'
DataPath=MainPath+'\\Output\\'
writer = ExcelWriter(DataPath+'YT_Trending_All.xlsx')
finaldf.to_excel(writer,'Sheet1')
writer.save()


#Pull video metadata for 50 each for top ten tags
#Selenium scraper
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
from datetime import datetime, date,timedelta, time as timepart 
import os.path
import random
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
AudioPath=r'C:\Users\manor\Desktop\Mainproject\Othervideos'
ffmpegexec=r'C:\Users\manor\Desktop\ffmpeg\bin\ffmpeg.exe'

import speech_recognition as sr
#import sphinxbase
#import pocketsphinx

import os
import subprocess
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as BeautifulSoup

tags_list = []
viewcount_list = []
title_list = []
audiotext_list = []
thumbnail_list = []
like_count_list = []
upload_date_list = []
youtubelinks_list = []
tag_searched_list = []

for tag in list(Top20Tags['ngram']):
    baseUrl=('https://www.youtube.com/')
    driver.get(baseUrl)
    wait = WebDriverWait(driver, 10)
    search_input = wait.until(EC.presence_of_element_located((By.ID, "search")))
    search_input.send_keys(tag)
    search_input.send_keys(u'\ue007')
    wait = WebDriverWait(driver, 10)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep((random.randrange(2, 5, 1))*0.1)
    elems = driver.find_element_by_id("content").find_elements_by_id("video-title")
    youtubealllinks= []

    for elem in elems:
        #print(elem)
        link=elem.get_attribute("href")
        #print(link)
        time.sleep((random.randrange(5, 9, 1))*0.1)
        youtubealllinks.append(link)
    #get top 5 links only from each list
    youtubealllinks = [x for x in youtubealllinks if x is not None]
    youtubelinks = youtubealllinks[:50]
    
    #print(youtubelinks)
    counter=1
    for link in youtubelinks:
        print(link)
        youtubelinks_list.append(link)
        testsrc=AudioPath+'\\'+tag+str(counter)
        testwav=AudioPath+'\\'+tag+str(counter)+'.wav'
        #options = {'outtmpl': '%(id)s'}  # save file as the YouTube ID
        options = {
          'format': 'bestaudio/best',
          'extractaudio' : True,  # only keep the audio
          #'audioformat' : "mp3",  # convert to mp3
          #'audioformat' : "
          'ignoreerrors': True,
          #'outtmpl': '%(id)s.%(ext)s',    # name the file the ID of the video
          'noplaylist' : True,    # only download single song, not playlist
          'outtmpl': testsrc+'.%(ext)s',
          #'postprocessors': [{
          #        'key': 'FFmpegExtractAudio',
          #        'preferredcodec': 'mp3',
          #        'preferredquality': '192',
          #        }]
        } 
        with youtube_dl.YoutubeDL(options) as ydl:
            #ydl.download([link])
            info_dict = ydl.extract_info(link, download=False)
        #Remove old file if its there
        try:
            os.remove(testwav)
        except:
            print("No File, its ok")
            
        Tagsdata=info_dict['tags']
        viewcount=info_dict['view_count']
        videotitle=info_dict['title']
        
        #Get tags
        page = urllib.request.urlopen(link)
        pagehtml = page.read()
        #create BeautifulSoup parse-able "soup"
        pagesoup = BeautifulSoup(pagehtml, "lxml")
        souptext = str(pagesoup)
        #get the src attribute from the video tag
        regexp = re.compile("(?<=keywords)(.*?)(?=\])")
        try:
            Tagsdataurl = str(regexp.search(souptext).group(1))
            Tagsdataurl = Tagsdataurl.replace(r'\"', '').replace('[','').replace(':','').split(",")
        except:
            Tagsdataurl = ''
        #print(Tagsdata)
        #Get Viewcount
        regexp = re.compile("(?<=stat view-count\">)(.*?)(?=\<)")
        try:
            viewcounttext=str(regexp.search(souptext).group(1)).replace(' views','')
            viewcounttext=viewcounttext.replace(',','')
        except:
            viewcounttext=viewcounttext
        try:
            viewcounturl=int(viewcounttext)
        except:
            viewcounturl=0
        
        #Get videotitle
        #videotitle =pagesoup.find(id='eow-title').find_all(text=True, recursive=False)
        #type(videotitle)
        textlist = [Member.get_text().strip() for Member in pagesoup.find_all(id='eow-title')]
        titletext = ''.join(textlist)
        #type(titletext)
        titletext = re.sub(r'[^\x00-\x7F]+',' ', titletext)
        videotitleurl=titletext.split('|')
            
        if not Tagsdata:
            Tagsdata=Tagsdataurl
        tags_list.append(Tagsdata)
        

        
        if not viewcount:
            viewcount=viewcounturl
        viewcount_list.append(viewcount)
        
        if not videotitle:
            videotitle=videotitleurl
        title_list.append(videotitle)
        
        thumbnail_list.append(info_dict['thumbnail'])
        like_count_list.append(info_dict['like_count'])
        upload_date_list.append(info_dict['upload_date'])
        tag_searched_list.append(tag)
        
        #Full downbloaded file name
        testfullsrc=testsrc+'.'+info_dict['ext']
        time.sleep((random.randrange(2, 6, 1))*0.1)
        counter = counter + 1

print("Ended Loop")
        
finaldf_extravideos = pd.DataFrame({'link': youtubelinks_list, 'video_title': title_list, 
                        'view_count': viewcount_list, 'like_count': like_count_list,
                        'upload_date': upload_date_list, 'thumbnail_link': thumbnail_list,
                        'tags': tags_list,'searched_tag': tag_searched_list})
#del title_list

    
#Put this data into Excel
from pandas import ExcelWriter

MainPath=r'C:\Users\manor\Desktop\Mainproject'
AudioPath=MainPath+'\\Trending\\'
DataPath=MainPath+'\\Output\\'
writer = ExcelWriter(DataPath+'YT_Top200TagsExtra.xlsx')
finaldf_extravideos.to_excel(writer,'Sheet1')
writer.save()
        
##End time of script
nowDTG = datetime.now().replace(microsecond=0)
print('Script ended at: ' + str(nowDTG))


#Merge all the below video tags files and create one excel file with binary count for tensorflow model
file_merge_list=['YT_Trending_All.xlsx', 'YT_TopVideos.xlsx', 'YT_WorstVideos.xlsx', 'YT_Top200TagsExtra.xlsx']

finaldf_allmerged=pd.DataFrame()
for x in file_merge_list:
    #print(x)
    FilePath=DataPath+x
    print(FilePath)
    xl = pd.ExcelFile(FilePath)
    finaldf_allmerged=finaldf_allmerged.append(xl.parse("Sheet1"))
   
finaldf_allmerged=finaldf_allmerged.reset_index(drop=True)

finaldf_allmerged['tags']=finaldf_allmerged['tags'].str.replace('[','').str.replace(']','').str.replace('\"','').str.replace("'",'').str.replace("nan",'')

#finaldf_allmerged['tags']=finaldf_allmerged['tags'].str.replace('[','').str .replace(']','').str.replace('\"','').str.replace("nan",'')


binarycounterlist=[]
for index, row in finaldf_allmerged.iterrows():
   listrow=str(row['tags']).split(",")
   listbinaryrow=[]
   #print(index)
   for i in listrow:
       print('searching for '+i)
       #print(type(i))
       #i=i.lstrip()
       if str(i).lower()=='nan':
           listbinaryrow.append(0)
       else:
           if i.lstrip() in list(Top200Tags['ngram'].values):
               print(str(i))
               print(list(Top200Tags['ngram'].str.replace("'",'')))
               print('YES')
               #time.sleep((random.randrange(30, 35, 1))*0.1)

               listbinaryrow.append(1)
           else:
               listbinaryrow.append(0)
               print(i)
               print(list(Top200Tags['ngram']))
               print('NO')
               #time.sleep((random.randrange(30, 35, 1))*0.1)
   listbinaryrow.sort(key=int, reverse=True)
   binarycounterlist.append(listbinaryrow)           

len(binarycounterlist)
finaldf_allmerged.shape

binarycounternewlist=[]
for item in binarycounterlist:
    print(len(item))
    if len(item) > 10:
        itemnew=item[:10]
        binarycounternewlist.append(itemnew)
    else:
        itemnew=item.extend([0] * (10 - len(item)))
        binarycounternewlist.append(itemnew)
    
        
finaldf_allmerged['binarycounter']=binarycounternewlist

writer = ExcelWriter(DataPath+'YT_Top200TagsBcounts.xlsx')
finaldf_allmerged.to_excel(writer,'Sheet1')
writer.save()

