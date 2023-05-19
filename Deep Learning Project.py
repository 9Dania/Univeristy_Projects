#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Project
# ## Movies Reviews Classification with Bidirectional LSTM and CNN 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from sklearn.model_selection import KFold

# re is used for cleaning the dataset 

import re
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords
from tensorflow.keras.layers import Embedding,Conv1D,LSTM ,Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score ,cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# ### 1- Load data & basic exploration 

# In[3]:


df = pd.read_csv('/Users/macbook/Desktop/IMDB Dataset.csv')


# In[4]:


df.head()


# In[5]:


df['sentiment'].value_counts().plot(kind = 'bar', color =[ 'pink', 'skyblue'])


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


#replace positive to 1, negative to 0
df['sentiment'].replace('positive',1, inplace = True)
df['sentiment'].replace('negative',0, inplace = True)
#df['sentiment'] = np.where(reviews['sentiment'] == 'positive', 1, 0)


# In[9]:


df.head()


# # Text Pre-processing 

# In[10]:


#cleane method DONE 👌 
stop_words = set(stopwords.words('english'))

def data_processing(text):
    
    text = re.sub('', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    #filtered_text = [w for w in text.lower() if not w in stop_words]
    #" ".join(filtered_text)
    words = [word.lower() for word in text.split() if word not in stop_words]
    new_text = " ".join(words)
    return new_text


# In[11]:


df.review = df['review'].apply(data_processing)


# In[12]:


duplicated_count = df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)


# In[13]:


df = df.drop_duplicates('review')


# In[14]:


df.head()


# In[15]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[16]:


df.review = df['review'].apply(lambda x: stemming(x))


# In[17]:


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count


# In[18]:


df['word count'] = df['review'].apply(no_of_words)
df.head()


# In[19]:


pos_reviews =  df[df.sentiment == 1]
pos_reviews.head()

neg_reviews =  df[df.sentiment == 0]
neg_reviews.head()


# In[20]:


X = df['review']
Y = df['sentiment']


# In[ ]:





# ## Wordcloud

# # Tokenization

# In[21]:


#tokenization is DONE 
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X)
if num_words is None:
    num_words = len(tokenizer.word_index)
#tokenizer.word_index


# In[22]:


# create numeric sequences representation of text
x_tokens = tokenizer.texts_to_sequences(X)


#number of tokens
num_tokens = [len(tokens) for tokens in x_tokens ]
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)


# # Vectorize the dataset

# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
X = vect.fit_transform(df['review'])


# # Machine Learning Models

# ## Second Algorithm

# In[55]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
_scoring = ['accuracy', 'precision', 'recall', 'f1']
result1 = cross_validate(logreg, X,Y, cv=10, scoring=_scoring, return_train_score=True)


# In[56]:


#store average of all 4 metrics of first model in train and test 
tr1_acc_avg = result1['train_accuracy'].mean() 
ts1_acc_avg = result1['test_accuracy'].mean()
tr1_prec_avg = result1['train_precision'].mean()
ts1_prec_avg= result1['test_precision'].mean()
tr1_recall = result1['train_recall'].mean()
ts1_recall = result1['test_recall'].mean()
tr1_f1 = result1['train_f1'].mean()
ts1_f1 = result1['test_f1'].mean()
#Standard deveation of all 4 metrics of first model in train and test 
tr1_acc_sd = result1['train_accuracy'].std()
ts1_acc_sd = result1['test_accuracy'].std()
tr1_prec_sd = result1['train_precision'].std()
ts1_prec_sd= result1['test_precision'].std()
tr1_recall_sd = result1['train_recall'].std()
ts1_recall_sd = result1['test_recall'].std()
tr1_f1_sd= result1['train_f1'].std()
ts1_f1_sd = result1['test_f1'].std()
    
from colorama import Fore
print(Fore.RED + 'Average and Standard deviation of 10 fold cross validation of Logistic Regression')
print(Fore.BLACK+"train Accuracy: {:.2%}".format(tr1_acc_avg),"\tStandard deviation: {:.2%}".format(tr1_acc_sd)) 
print("test Accuracy:  {:.2%}".format(ts1_acc_avg), "\tStandard deviation: {:.2%}".format(ts1_acc_sd)) 
print('-------------------------------------------')
print("train Precision:{:.2%}".format(tr1_prec_avg), "\tStandard deviation: {:.2%}".format(tr1_prec_sd ))
print("test Precision: {:.2%}".format(ts1_prec_avg), "\tStandard deviation: {:.2%}".format(ts1_prec_sd ))
print('-------------------------------------------')
print("train Recall: {:.2%}".format(tr1_recall), "\tStandard deviation: {:.2%}".format(tr1_recall_sd  ))
print("test Recall:  {:.2%}".format(ts1_recall),  "\tStandard deviation:  {:.2%}".format(ts1_recall_sd  ))
print('-------------------------------------------')
print("train F1: {:.2%}".format(tr1_f1 ), "\tStandard deviation: {:.2%}".format(tr1_f1_sd ))
print("test F1:  {:.2%}".format(ts1_f1 ),  "\tStandard deviation: {:.2%}".format(ts1_f1_sd))
    


# ## Third Algorithm

# In[58]:


from sklearn.svm import LinearSVC
svc = LinearSVC()
_scoring = ['accuracy', 'precision', 'recall', 'f1']
result2 = cross_validate(svc, X,Y, cv=10, scoring=_scoring, return_train_score=True)


# In[59]:


#store average of all 4 metrics of first model in train and test 
tr2_acc_avg = result2['train_accuracy'].mean() 
ts2_acc_avg = result2['test_accuracy'].mean()
tr2_prec_avg = result['train_precision'].mean()
ts2_prec_avg= result2['test_precision'].mean()
tr2_recall = result2['train_recall'].mean()
ts2_recall = result2['test_recall'].mean()
tr2_f1 = result2['train_f1'].mean()
ts2_f1 = result2['test_f1'].mean()
#Standard deveation of all 4 metrics of first model in train and test 
tr2_acc_sd = result2['train_accuracy'].std()
ts2_acc_sd = result2['test_accuracy'].std()
tr2_prec_sd = result2['train_precision'].std()
ts2_prec_sd= result2['test_precision'].std()
tr2_recall_sd = result2['train_recall'].std()
ts2_recall_sd = result2['test_recall'].std()
tr2_f1_sd= result2['train_f1'].std()
ts2_f1_sd = result2['test_f1'].std()

from colorama import Fore
print(Fore.RED + 'Average and Standard deviation of 10 fold cross validation of LinearSVC')
print(Fore.BLACK+"train Accuracy: {:.2%}".format(tr2_acc_avg),"\tStandard deviation: {:.2%}".format(tr2_acc_sd)) 
print("test Accuracy:  {:.2%}".format(ts2_acc_avg), "\tStandard deviation: {:.2%}".format(ts2_acc_sd)) 
print('-------------------------------------------')
print("train Precision:{:.2%}".format(tr2_prec_avg), "\tStandard deviation: {:.2%}".format(tr2_prec_sd ))
print("test Precision: {:.2%}".format(ts2_prec_avg), "\tStandard deviation: {:.2%}".format(ts2_prec_sd ))
print('-------------------------------------------')
print("train Recall: {:.2%}".format(tr2_recall), "\tStandard deviation: {:.2%}".format(tr2_recall_sd  ))
print("test Recall:  {:.2%}".format(ts2_recall),  "\tStandard deviation:  {:.2%}".format(ts2_recall_sd  ))
print('-------------------------------------------')
print("train F1: {:.2%}".format(tr2_f1 ), "\tStandard deviation: {:.2%}".format(tr2_f1_sd ))
print("test F1:  {:.2%}".format(ts2_f1 ),  "\tStandard deviation: {:.2%}".format(ts2_f1_sd))
    
    


# # Padding

# In[130]:


#padding 
pad = 'pre'
x_pad = pad_sequences(x_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
#x_pad.shape


# # First Model: LSTM

# In[131]:


def create_model():
    # create model
    model = Sequential()
    embedding_size = 8
    model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
    lstm_out=32
    #2nd improvement : model.add(Bidirectional(LSTM(lstm_out)))
    #3rd improvment: model.add(Conv1D(filters=32, kernel_size=3, padding='pad', activation='relu'))
    model.add(LSTM(128,activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# In[132]:


scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

# define model and kFold
estimator = KerasClassifier(build_fn=create_model, epochs=1, batch_size=64,     verbose=0)
kfold = KFold(n_splits=10, shuffle=True)

#define our accuracy metric
metrics = ['accuracy', 'precision', 'recall', 'f1']

# fit to model
result1 = cross_validate(estimator, x_pad, Y, scoring=metrics, cv=kfold , return_train_score=True)


# In[133]:


#store average of all 4 metrics of first model in train and test 
train1_acc_avg = result['train_accuracy'].mean() 
test1_acc_avg = result['test_accuracy'].mean()
train1_prec_avg = result['train_precision'].mean()
test1_prec_avg= result['test_precision'].mean()
train1_recall = result['train_recall'].mean()
test1_recall = result['test_recall'].mean()
train1_f1 = result['train_f1'].mean()
test1_f1 = result['test_f1'].mean()


# In[134]:


from colorama import Fore
print(Fore.RED + 'Average of 10 fold cross validation of First DL model (LSTM)')
print(Fore.BLACK+"train Accuracy: {:.2%}".format(train1_acc_avg)) 
print("test Accuracy: {:.2%}".format(test1_acc_avg)) 
print('-------------------------------------------')
print("train Precision: {:.2%}".format(train1_prec_avg ))
print("test Precision: {:.2%}".format(test1_prec_avg ))
print('-------------------------------------------')
print("train Recall: {:.2%}".format(train1_recall ))
print("test Recall: {:.2%}".format(test1_recall ))
print('-------------------------------------------')
print("train F1: {:.2%}".format(train1_f1 ))
print("test F1: {:.2%}".format(test_f1 ))


# # 2nd improvment: Bidirectional LSTM 
# 
# ### add Bidirectional layer rather than simple LSTM

# In[155]:


def create_2model():
    # create model
    model = Sequential()
    embedding_size = 8
    model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
    model.add(Dropout(0.2))
    lstm_out=32
    #2nd improvement : model.add(Bidirectional(LSTM(lstm_out)))
    #3rd improvment: model.add(Conv1D(filters=32, kernel_size=3, padding='pad', activation='relu'))
    model.add(LSTM(lstm_out , activation='relu')) 
    model.add(Dropout(0.2))
    
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# In[156]:


estimator = KerasClassifier(build_fn=create_2model, epochs=2, batch_size=64,     verbose=0)
result2 = cross_validate(estimator, x_pad, Y, scoring=metrics, cv=kfold , return_train_score=True)


# In[1]:


#store average of all 4 metrics of first model in train and test 
train2_acc_avg = result2['train_accuracy'].mean() 
test2_acc_avg = result2['test_accuracy'].mean()
train2_prec_avg = result2['train_precision'].mean()
test2_prec_avg= result2['test_precision'].mean()
train2_recall = result2['train_recall'].mean()
test2_recall = result2['test_recall'].mean()
train2_f1 = result2['train_f1'].mean()
test2_f1 = result2['test_f1'].mean()


# In[ ]:


result.summary() 


# In[149]:


result2.summary() 


# In[ ]:


from colorama import Fore
print(Fore.RED + 'Average of 10 fold cross validation of 1st improvement of DL model (Bidirectional LSTM)')
print(Fore.BLACK+"train Accuracy: {:.2%}".format(train2_acc_avg)) 
print("test Accuracy: {:.2%}".format(test2_acc_avg)) 
print('-------------------------------------------')
print("train Precision: {:.2%}".format(train2_prec_avg ))
print("test Precision: {:.2%}".format(test2_prec_avg ))
print('-------------------------------------------')
print("train Recall: {:.2%}".format(train2_recall ))
print("test Recall: {:.2%}".format(test2_recall ))
print('-------------------------------------------')
print("train F1: {:.2%}".format(train2_f1 ))
print("test F1: {:.2%}".format(test2_f1 ))


# # second Improvement of deep learning algorithm using CNN layer + Bidirectional LSTM (changed )

# In[147]:


def create_3model():
    # create model
    model = Sequential()
    embedding_size = 8
    model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
    lstm_out=32
    #2nd improvement : model.add(Bidirectional(LSTM(lstm_out)))
    #3rd improvment: model.add(Conv1D(filters=32, kernel_size=3, padding='pad', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128,activation='relu', dropout=0.2, recurrent_dropout=0.2)) 
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# In[148]:



estimator = KerasClassifier(build_fn=create_3model, epochs=1, batch_size=64,     verbose=0)
result3 = cross_validate(estimator, x_pad, Y, scoring=metrics, cv=kfold , return_train_score=True)


# In[150]:


#store average of all 4 metrics of first model in train and test 
train3_acc_avg = result3['train_accuracy'].mean() 
test3_acc_avg = result3['test_accuracy'].mean()
train3_prec_avg = result3['train_precision'].mean()
test3_prec_avg= result3['test_precision'].mean()
train3_recall = result3['train_recall'].mean()
test3_recall = result3['test_recall'].mean()
train3_f1 = result3['train_f1'].mean()
test3_f1 = result3['test_f1'].mean()


# In[151]:


from colorama import Fore
print(Fore.RED + 'Average of 10 fold cross validation of 1st improvement of DL model (LSTM)')
print(Fore.BLACK+"train Accuracy: {:.2%}".format(train3_acc_avg)) 
print("test Accuracy: {:.2%}".format(test3_acc_avg)) 
print('-------------------------------------------')
print("train Precision: {:.2%}".format(train3_prec_avg ))
print("test Precision: {:.2%}".format(test3_prec_avg ))
print('-------------------------------------------')
print("train Recall: {:.2%}".format(train3_recall ))
print("test Recall: {:.2%}".format(test3_recall ))
print('-------------------------------------------')
print("train F1: {:.2%}".format(train3_f1 ))
print("test F1: {:.2%}".format(test3_f1 ))







