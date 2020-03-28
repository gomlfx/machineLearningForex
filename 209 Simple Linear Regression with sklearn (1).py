#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# In[2]:


data_pre = pd.read_csv('EURUSD_M1_197101040000_202003272153.csv', delimiter='\t')
data = data_pre.rename(columns={'<DATE>':'date','<TIME>':'time','<OPEN>':'open','<HIGH>':'high','<LOW>':'low','<CLOSE>':'close','<TICKVOL>':'tickvol','<VOL>':'volume','<SPREAD>':'spread'})
data['index']=np.arange(len(data))
data = data.tail(592)
data.tail()


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


x = data['index']
y = data['open']


# In[6]:


x.shape


# In[7]:


y.shape


# In[8]:


x_matrix = x.values.reshape(-1,1)
x_matrix.shape


# In[9]:


reg = LinearRegression(n_jobs=2)


# In[10]:


reg.fit(x_matrix,y)


# In[11]:


reg.score(x_matrix,y)


# In[12]:


reg.coef_


# In[13]:


reg.intercept_


# In[14]:


#use last 2 points to predict next point
new_data = pd.DataFrame(data = [691200,691201], columns=['index'])
new_data


# In[15]:


reg.predict(new_data)


# In[16]:


plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x,yhat, lw=4, c='orange', label = 'regression line')
plt.xlabel('index', fontsize = 20)
plt.ylabel('open', fontsize = 20)
plt.show()

