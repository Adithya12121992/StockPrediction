# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:04:41 2020

@author: Guest_User
"""
import tweepy
import csv
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

analyzer = SentimentIntensityAnalyzer()
neg=0
neu=0
pos=0
pos1=0
neg1=0
neu1=0
stop_words = set(stopwords.words('english'))
ps=PorterStemmer()
filehandler =open("aapltwitter_output.csv","w",encoding="utf-8")
filehandler.writelines("{0},{1},{2},{3},{4}\n".format('date','tweet','sentiment with textblob','sentiment with vader','polarity with vader'))
with open("previousStockDataTSLA.csv", encoding ='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',') # splitting the data with "," since it is .csv 
    next(csv_reader)
    for row in csv_reader: # iterating through each and every line in the file
        #row=re.sub("!|/|:", "", row[1])
        lines=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', row[1])
        lines=re.sub('[^A-Za-z0-9]+', ' ', lines)
        tokens=word_tokenize(lines)
        words = [w.lower() for w in tokens if w.isalpha()] 
        tokens = [w for w in words if not w in stop_words] 
        stems = [] 
        for t in tokens:    
            stems.append(ps.stem(t))
        row=[i for i in row if not i in stop_words]
        for t in tokens:    
            stems.append(ps.stem(t))
        frequency_dist = nltk.FreqDist(stems) 
        seperated_wc = ' '.join(list(stems))
        analysis = TextBlob(seperated_wc)        
        if analysis.sentiment[0] >0:
            sentiment1 = 'positive'
            pos+=1
        elif analysis.sentiment[0] <0: 
            sentiment1 = 'negative'
            neg+=1
        else:
            sentiment1 = 'neutral'
            neu+=1
        vs = analyzer.polarity_scores(lines)
        if vs['compound']>=0.5:
            sentiment2 = 'positive'
            pos1+=1
            score1=analyzer.polarity_scores(lines)
        elif vs['compound'] <=-0.5: 
            sentiment2 = 'negative'
            neg1+=1
            score1=analyzer.polarity_scores(lines)            
        else:
            sentiment2 = 'neutral'
            score1=analyzer.polarity_scores(lines)
            neu1+=1

        filehandler.writelines("{0},{1},{2},{3},{4}\n".format(row[2],lines,sentiment1,sentiment2,score1)) # writing the output to a file
filehandler.close()

labels = 'Positive', 'Neutral', 'Negative'
sizes = [pos, neu, neg]
explode = (0, 0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("sentiment analysis with text blob")
plt.show()

labels = 'Positive', 'Neutral', 'Negative'
sizes = [pos1, neu1, neg1]
explode = (0, 0.1, 0)
fig2, ax2 = plt.subplots()
ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')
plt.title("sentiment analysis with vader sentiment analyser")
plt.show()


        
       
       