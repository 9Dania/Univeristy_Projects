#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[2]:


get_ipython().system('pip install wordcloud')


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")


# In[ ]:





# In[9]:


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count


# In[10]:


df['word count'] = df['review'].apply(no_of_words)


# In[11]:


df.head()


# In[12]:


df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 0, inplace=True)


# In[13]:


df.head(10)


# In[15]:


def data_processing(text):
    text= text.lower()
    text = re.sub('', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[16]:


df.review = df['review'].apply(data_processing)


# In[18]:


duplicated_count = df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)


# In[19]:


df = df.drop_duplicates('review')


# In[20]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[21]:


df.review = df['review'].apply(lambda x: stemming(x))


# In[22]:


df['word count'] = df['review'].apply(no_of_words)
df.head()


# In[24]:


pos_reviews =  df[df.sentiment == 1]
pos_reviews.head()


# In[37]:





# In[38]:


X = df['review']
Y = df['sentiment']


# In[ ]:




