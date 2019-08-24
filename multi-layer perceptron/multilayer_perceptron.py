#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv('Advertising.csv')


# In[3]:


data.head()  #只显示前五组数据


# In[6]:


plt.scatter(data.TV, data.sales)


# In[7]:


plt.scatter(data.radio, data.sales)


# In[8]:


plt.scatter(data.newspaper, data.sales)


# In[9]:


x = data.iloc[:,1:-1] #取所有行，除去第一列和最后一列的数据
y = data.iloc[:,-1]   #取最后一列数据


# In[17]:


model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)]
                           )
                            
# input_shape=(3,)等价于input_dim=3 初始化只需要在第一层定义,3组一元数据
# 列表[]直接定义，省去add，python基础- -
# Dense层 ax1+ bx2+ cx3+... + b'
# 激活函数‘relu’


# In[18]:


model.summary()


# In[19]:


#多层感知器，40 = 10 * （3（x）+ b）， 11= 10* w1 +b 


# In[27]:


model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs = 1000)


# In[28]:


test = data.iloc[:10,1:-1]  #前十个模型预测值


# In[29]:


model.predict(test)


# In[31]:


test = data.iloc[:10, -1]


# In[32]:


test


# In[ ]:




