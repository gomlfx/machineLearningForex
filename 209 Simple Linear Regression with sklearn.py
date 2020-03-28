#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# In[606]:


data_pre = pd.read_csv('EURUSD_Weekly_197101030000_202003150000.csv', delimiter='\t')
data = data_pre.rename(columns={'<DATE>':'date','<TIME>':'time','<OPEN>':'open','<HIGH>':'high','<LOW>':'low','<CLOSE>':'close','<TICKVOL>':'tickvol','<VOL>':'volume','<SPREAD>':'spread'})
data['index']=np.arange(len(data))
data = data.tail(592)
data.tail()


# In[607]:


data


# In[608]:


data.describe()


# In[609]:


x = data['index']
y = data['open']


# In[610]:


x.shape


# In[611]:


y.shape


# In[612]:


x_matrix = x.values.reshape(-1,1)
x_matrix.shape


# In[613]:


reg = LinearRegression(n_jobs=2)


# In[614]:


reg.fit(x_matrix,y)


# In[615]:


reg.score(x_matrix,y)


# In[616]:


reg.coef_


# In[617]:


reg.intercept_


# In[618]:


#use last 2 points to predict next point
new_data = pd.DataFrame(data = [691200,691201], columns=['index'])
new_data


# In[619]:


reg.predict(new_data)


# In[620]:


plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x,yhat, lw=4, c='orange', label = 'regression line')
plt.xlabel('index', fontsize = 20)
plt.ylabel('open', fontsize = 20)
plt.show()


# In[ ]:


1.068

