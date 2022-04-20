#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
import numpy as np
import surprise
from surprise import Reader, Dataset
# It is to specify how to read the data frame.
reader = Reader(rating_scale=(1,5))

from scipy import sparse


# In[2]:


import os
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from collections import defaultdict


# In[3]:


df=pd.read_csv('train.dat', sep='\t', header=None)
df.columns=['user', 'movie','rating','timestamp']


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


reader = Reader(rating_scale=(0.5, 5.0))
data_set = Dataset.load_from_df(df[['user', 'movie', 'rating']], reader)


# In[14]:


#Reference from: https://towardsdatascience.com/movie-recommender-system-part-1-7f126d2f90e2

param_grid = {'n_factors': [25, 30, 35, 40], 'n_epochs': [15, 20, 25], 'lr_all': [0.001, 0.003, 0.005, 0.008],
              'reg_all': [0.08, 0.1, 0.15]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data_set)
algo = gs.best_estimator['rmse']
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

#Assigning values
t = gs.best_params
factors = t['rmse']['n_factors']
epochs = t['rmse']['n_epochs']
lr_value = t['rmse']['lr_all']
reg_value = t['rmse']['reg_all']


# In[16]:


##{'n_factors': 35, 'n_epochs': 25, 'lr_all': 0.008, 'reg_all': 0.08}


# ## Training and Testing

# In[63]:


train_data, test_data = train_test_split(data_set, test_size=0.20, random_state=42)

reg = SVD(n_factors=35, n_epochs=30, lr_all=0.008, reg_all=0.05)
reg.fit(train_data)


# In[64]:


preds = reg.test(test_data)
accuracy.rmse(preds)


# In[28]:


preds[0:5]


# ## Test Data

# In[29]:


df_test=pd.read_csv('test.dat', sep='\t', header=None)
df_test.columns=['user', 'movie']
df_test['rating']=0


# In[30]:


df_test


# In[31]:


reader = Reader(rating_scale=(0.5, 5.0))
data_test = Dataset.load_from_df(df_test[['user', 'movie','rating']], reader)


# In[65]:


train_data = data_set.build_full_trainset() 

reg = SVD(n_factors=35, n_epochs=30, lr_all=0.008, reg_all=0.05)
reg.fit(train_data)


# In[66]:


data_set.df.to_numpy()


# In[67]:


preds_train = reg.test(data_set.df.to_numpy())
accuracy.rmse(preds_train)


# In[68]:


testset = [data_test.df.loc[i].to_list() for i in range(len(data_test.df))]


# In[69]:


predictions = reg.test(testset)


# In[70]:


pred=[]
for prediction in predictions:
    pred.append(prediction[3])


# In[71]:


len(pred)


# In[72]:


submission_df=pd.DataFrame(pred)


# In[73]:


submission_df


# In[74]:


submission_df.to_csv('submission2.dat', index=False, header=False)


# In[55]:


temp=pd.read_csv('format.dat')


# In[56]:


temp


# In[59]:


submission=pd.read_csv('submission1.dat')


# In[60]:


submission


# In[ ]:




