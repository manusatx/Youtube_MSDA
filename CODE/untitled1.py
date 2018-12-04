# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:09:47 2018

@author: manor
"""

import requests
response = requests.get("https://www.youtube.com/watch?v=r7nzwVWWGKg")
textpage = response.content
print(textpage)
import re
pattern = re.compile("\\"keywords\\":[")


from bs4 import BeautifulSoup as bs
import requests
base = "https://www.youtube.com/watch?v=r7nzwVWWGKg"
r = requests.get(base)
page = r.text
soup=bs(page,'html.parser')
print(soup)

text=soup.find('keywords')
print(text)

soup.g

vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})

import urllib
url = 'https://www.youtube.com/watch?v=r7nzwVWWGKgr'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36'

# header variable
headers = { 'User-Agent' : user_agent }

# creating request
req = urllib.Request(url, None, headers)

# getting html
html = urllib.urlopen(req).read()




def getsource(url):
    req=urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}) #sends GET request to URL
    uClient=urllib.request.urlopen(req)
    page_html=uClient.read() #reads returned data and puts it in a variable
    uClient.close() #close the connection
    page_soup=BeautifulSoup(page_html,"html.parser")
    return [page_soup, page_html]

datad = getsource(url)

#print(datad[0])

soup = datad[0]

soup.getText()

soup.pattern(pattern)

regexp = re.compile("keywords(.*)$")
print regexp.search(soup).group(1)

import re
import urllib
from bs4 import BeautifulSoup as BS

url = '''https://www.youtube.com/watch?v=6dChKENkLLk'''
#open and read page
page = urllib.request.urlopen(url)
html = page.read()
#create BeautifulSoup parse-able "soup"
soup = BS(html, "html.parser")
souptext = str(soup)
#get the src attribute from the video tag
regexp = re.compile("(?<=keywords)(.*?)(?=\])")
#print(str(regexp.search(souptext).group(1)))
  Tagsdata = str(regexp.search(souptext).group(1))
Tagsdata = Tagsdata.replace(r'\"', '').replace('[','').replace(':','').split(",")
print(Tagsdata)

regexp = re.compile("(?<=watch-view-count\">)(.*?)(?=\<)")
viewcountstr=str(regexp.search(souptext).group(1)).replace(' views','')
try:
    viewcountstr=viewcountstr.replace(',','')
except:
    viewcountstr=viewcountstr
viewcount=int(viewcountstr)


souptext=str(souptext.encode("utf-8", 'ignore'))

type(souptext)
soup.find('eow-title')

videotitle=soup.find_all(id='eow-title')
videotitle.text()

textlist = [Member.get_text().strip() for Member in soup.find_all(id='eow-title')]
titletext = ''.join(textlist)
#type(titletext)
titletext = re.sub(r'[^\x00-\x7F]+',' ', titletext)
videotitle=titletext.split('|')

 if not Member.get_text().encode("utf-8").startswith("\xe2")]


.find_all(text=True, recursive=False)
videotitle.replace('\\n','')
type(videotitle)

str(videotitle).encode('ascii','ignore')


unicode.join(u'\n',map(unicode,videotitle))

videotitle.pattern

encode('ascii','ignore')
.normalize('NFKC', p.decode()).encode('ascii','ignore')


#decode('utf-8', 'ignore')
regexp = re.compile("(?<=eow-title\">)(.*?)(?=\"<)")

titlestr=str(regexp.search(souptext).group(1))

re.sub(r'[^\w.]', '', souptext) 


try:
    viewcountstr=viewcountstr.replace(',','')
except:
    viewcountstr=viewcountstr
viewcount=int(viewcountstr)



from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

vector_u = CountVectorizer(ngram_range = (1,1), analyzer = 'word', stop_words=stopwords.words('english'))
unigrams = vector_u.fit(Tagsdata).get_feature_names()

options = {
  'format': 'bestaudio/best',
  'extractaudio' : True,  # only keep the audio
  'audioformat' : "mp3",  # convert to mp3
  #'outtmpl': '%(id)s.%(ext)s',    # name the file the ID of the video
  'noplaylist' : True,    # only download single song, not playlist
  'outtmpl': testsrc+'.%(ext)s',
} 
link=r'https://www.youtube.com/watch?v=ll4PiBEWFvI'
import youtube_dl
with youtube_dl.YoutubeDL(options) as ydl:
    info_dict = ydl.extract_info(link, download=True)
    
info_dict['tags']
info_dict['view_count']
info_dict['title']
info_dict['thumbnail']
info_dict['like_count']

info_dict['ext']

seconds = "4"
subprocess.call([ffmpegexec, '-i', r'C:\Users\manor\Desktop\Mainproject\Trending\Trending3.m4a', '-ss', seconds, testwav])
    
Top50Tags["count"] = pd.to_numeric(Top50Tags["count"])

Top50Tags["count"] = Top50Tags["count"].astype(int)


Top50Tagsdict=Top50Tags.set_index('ngram').T.to_dict('list')

.set_index('ID').T.to_dict('list')




binarycounterlist=[]
for index, row in finaldf_allmerged.iterrows():
   listrow=str(row['tags']).split(",")
   listbinaryrow=[]
   print(index)
   for i in listrow:
       print(i)
       print('manu')
       if str(i).lower()=='nan':
           listbinaryrow.append(0)
       else:
           if i in list(Top200Tags['ngram']):
               print(str(i).lower())
               print('1')
               print(str(list(Top50Tags['ngram'])) +'           ' + 'YES')
               time.sleep((random.randrange(18, 25, 1))*0.1)
           else:
               print('0')
               print(str(list(TopTenTags['ngram'])) +'           ' + 'NO')
               time.sleep((random.randrange(12, 25, 1))*0.1)
           

       if any(i in s for s in TopTenTags):
           print(i)
           print('1')
       else:
           print('0')   

i = ' funny'
if i.lstrip() in list(Top50Tags['ngram'].values):
    print(str(i)) 
           
item_list = ["Non-Tradable Ubersaw", "Screamin' Eagle", "'Non-Craftable Spy-cicle"]
not_allowed = {"Non-Tradable", "Non-Craftable"}   
filtered = [1 for i in item_list if any(stop in i for stop in not_allowed)]



if any("abc" in s for s in TopTenTags):
    

list(finaldf_allmerged['tags'][:1])
allmergedlisttags=list(finaldf_allmerged['tags'].str.replace('[','').str.replace(']','').str.replace('\"','').str.replace("'",''))
allmergedlisttagsstr=','.join(str(e) for e in allmergedlisttags)
allmergedtagslist=listtagsstr.split(",")

