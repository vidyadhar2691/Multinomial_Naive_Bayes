# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:57:54 2015

@author: Vidyadhar
"""

import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import operator 
import math
import csv


tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
stopset = set(sorted(stopwords.words('english')))

path="train"
class_list= os.listdir(path)


#Create Vocab List
def vocab_list():
    print ('Preparing Vocabulary...')
    vocab=[];  #list to store to maintain vocabulary 
    for x in class_list:
        class_files=os.listdir(path+"\\"+x)
        for y in class_files:        
            f=open(path+"\\"+x+"\\"+y,"r")
            tokens=tokenizer.tokenize(f.read().lower())            
            vocab.extend(tokens)
            f.close()
    vocab=list(set(vocab))
    filtered_words = [stemmer.stem(x) for x in vocab if x not in stopset]
    return filtered_words
    
    
#Calculate Total Docs  
def count_docs():
    total_docs=0
    for x in class_list:
        temp=os.listdir(path+"\\"+x)
        total_docs=total_docs+len(temp)
    return total_docs
    
#Total Docs belong to a particular class
def count_class_docs(var_class):
        temp=os.listdir(path+"\\"+var_class)
        return len(temp)

#Conatacting all docs to get a single text of a particular classs
def concat_doc(var_class):
    doc_list=os.listdir(path+"\\"+var_class)
    text=[]
    for y in doc_list:        
            f=open(path+"\\"+var_class+"\\"+y,"r")
            tokens=tokenizer.tokenize(f.read().lower())            
            text.extend(tokens)
            f.close()
    text_list = [stemmer.stem(x) for x in text if x not in stopset]
    return text_list
            


#Method to Train Data
def train_data():
    print ('Training Multinomial Naive Bayes Classifier...')
    vocab=vocab_list()
    total_docs=count_docs()
    prior={}
    cond_prob={}
    for x in class_list:
        cond_prob[x]={}
        class_count=count_class_docs(x)
        prior[x]=class_count/total_docs
        text=concat_doc(x)
        for y in vocab:
            token_count=text.count(y)
            cond_prob[x][y]=(token_count+1)/(len(text)+len(vocab))
    return vocab,prior,cond_prob
            
            
#Method to implement Data   
def MultinomialNaiveBayes(doc,prior,cond_prob,vocab):
    f=open("test"+"\\"+doc,"r")
    tokens=tokenizer.tokenize(f.read().lower())
    token_list = [stemmer.stem(x) for x in tokens if x not in stopset]
    doc_token1=list(set(token_list))
    doc_token=[x for x in doc_token1 if x in vocab] 
    class_value={}
    for c in class_list:
        class_value[c]=math.log(prior[c])
        for d in doc_token:
            class_value[c]=class_value[c]+math.log(cond_prob[c][d])
    return max(class_value.items(), key=operator.itemgetter(1))[0]
        
        
    
        



#Main Function
vocab,prior,cond_prob=train_data()

test_path="test"
test_docs=os.listdir(test_path)
final_list={}
print ('Classifying the test documents')
for doc in test_docs:
    doc_class=MultinomialNaiveBayes(doc,prior,cond_prob,vocab)
    final_list[doc]=doc_class

print ('Writing the results to a CSV file')
writer = csv.writer(open('result.csv', 'w', newline=""))
for key, value in final_list.items():
   writer.writerow([key, value])

print ('Finish');





