# -*- coding: utf-8 -*-
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
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as BeautifulSoup
import sys
import pandas as pd
import numpy as np

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
#Persons = ['Ray Dalio interview', 'Kenneth Arrow' , 'James P. Gorman interview', 'Jamie Dimon interview' ]
  
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

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import youtube_dl
AudioPath=r'C:\Users\manor\Desktop\Mainproject\OtherVideos'
ffmpegexec=r'C:\Users\manor\Desktop\ffmpeg\bin\ffmpeg.exe'

#import sphinxbase
#import pocketsphinx

import os
import subprocess

baseUrl=('https://www.youtube.com/')    
searchterm= 'Top Videos'
driver.get(baseUrl)
wait = WebDriverWait(driver, 10)
search_input = wait.until(EC.presence_of_element_located((By.ID, "search")))
search_input.send_keys(searchterm)
wait = WebDriverWait(driver, 10)
search_input.send_keys(u'\ue007')
wait = WebDriverWait(driver, 10)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep((random.randrange(10, 20, 1))*0.1)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep((random.randrange(10, 20, 1))*0.1) 
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep((random.randrange(10, 20, 1))*0.1)
elems = driver.find_elements_by_id("video-title")
youtubealllinks= []
for elem in elems:
    #print(elem)
    link=elem.get_attribute("href")
    #print(link)
    time.sleep((random.randrange(5, 9, 1))*0.1)
    youtubealllinks.append(link)

#get top 50 links only from each list
youtubelinks = [x for x in youtubealllinks if x is not None]
#youtubelinks = youtubelinks[:50]
#print(youtubelinks)

tags_list = []
viewcount_list = []
title_list = []
audiotext_list = []
thumbnail_list = []
like_count_list = []
upload_date_list = []
#Loop through the links
counter=1
for link in youtubelinks:
    print(link)
    
    testsrc=AudioPath+'\\'+'TopVideo'+str(counter)
    testwav=AudioPath+'\\'+'TopVideo'+str(counter)+'.wav'
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
        info_dict = ydl.extract_info(link, download=True)
   
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
    viewcounturl=int(viewcounttext)
    
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
    
    #Full downbloaded file name
    testfullsrc=testsrc+'.'+info_dict['ext']
    
    #Remove old file if its there
    try:
        os.remove(testwav)
    except:
        print("No File, its ok")
    #call ffmpeg to convert to wave
    seconds = "4"
    #subprocess.call([ffmpegexec, '-i', testsrc, '-ss', seconds, testwav])
    p = subprocess.Popen([ffmpegexec, '-i', testfullsrc, '-ss', seconds, testwav])
    if p.wait() != 0:
        print("There was an error")
    time.sleep((random.randrange(5, 9, 1))*0.1)
    counter=counter+1
       
##End time of script
nowDTG = datetime.now().replace(microsecond=0)
print('Script ended at: ' + str(nowDTG))

finaldf = pd.DataFrame({'link': youtubelinks, 'video_title': title_list, 
                        'view_count': viewcount_list, 'like_count': like_count_list,
                        'upload_date': upload_date_list, 'thumbnail_link': thumbnail_list,
                        'tags': tags_list})
del title_list

#Put this data into Excel
from pandas import ExcelWriter

MainPath=r'C:\Users\manor\Desktop\Mainproject'
AudioPath=MainPath+'\\OtherVideos\\'
DataPath=MainPath+'\\Output\\'
writer = ExcelWriter(DataPath+'YT_TopVideos.xlsx')
finaldf.to_excel(writer,'Sheet1')
writer.save()