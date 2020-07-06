#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('twitter_sentiments.csv')
data.head()


# In[3]:


train, test = train_test_split(data, test_size = 0.2, stratify = data['label'], random_state=21)


# In[4]:


train.shape, test.shape


# In[5]:


tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer.fit(train.tweet)


# In[6]:


train_idf = tfidf_vectorizer.transform(train.tweet)
test_idf  = tfidf_vectorizer.transform(test.tweet)


# In[7]:


model_LR = LogisticRegression()
model_LR.fit(train_idf, train.label)
predict_train = model_LR.predict(train_idf)
predict_test = model_LR.predict(test_idf)
f1_score(y_true= train.label, y_pred= predict_train)
f1_score(y_true= test.label, y_pred= predict_test)


# In[8]:


pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=1000,
                                                      stop_words= ENGLISH_STOP_WORDS)),
                            ('model', LogisticRegression())])                          
pipeline.fit(train.tweet, train.label)


# In[9]:


#testing the pipeline
text = ["@user lets fight against  #love #peace"]
pipeline.predict(text)


# In[10]:


from joblib import dump                #saving the pipeline 
dump(pipeline, filename="text_classification.joblib")


# In[11]:


#using saved file
from joblib import load
text = ["Virat Kohli, AB de Villiers set to auction their 'Green Day' kits from 2016 IPL match to raise funds"]
pipeline = load("text_classification.joblib")
pipeline.predict(text)


# In[ ]:




