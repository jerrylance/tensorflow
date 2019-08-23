#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data = pd.read_csv('credit-a.csv', header = None) #这个表没有表头，加个参数header = None,增加表头


# In[11]:


data.head()


# In[13]:


data.iloc[:,-1].value_counts()   #value_counts() 计数，结果为1的数据357个，-1的数据296个


# In[17]:


x = data.iloc[:,:-1]
y = data.iloc[:,-1].replace(-1,0) #将全部-1替换为0，方便之后使用sigmoid函数


# In[18]:


model = tf.keras.Sequential()


# In[19]:


model.add(tf.keras.layers.Dense(4, input_shape = (15,), activation = 'relu'))
model.add(tf.keras.layers.Dense(4, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))


# In[21]:


model.summary()  #简单网络4层就够了


# In[24]:


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc']) # metrics函数可以计算很多，如loss等，这里'acc'是准确度


# In[26]:


history = model.fit(x, y, epochs = 1000)


# In[28]:


history.history.keys()  #  .histroy.keys()记录的关键变化


# In[33]:


plt.plot(history.epoch, history.history.get('loss'))  #或者用range(1000)


# In[34]:


plt.plot(range(1000), history.history.get('acc'))


# In[51]:


test = data.iloc[:10,:-1]  #前十个  可以把test换成x
model.predict(test)


# In[52]:


test = data.iloc[:10,-1]  #可以把test换成y
test


# In[ ]:




