#!/usr/bin/env python
# coding: utf-8

# In[1]:


#f(x) = kx + b ， f(x)-b 为弹簧增量长度，受力为x


# In[9]:


import tensorflow as tf


# In[12]:


import pandas as pd


# In[15]:


data = pd.read_csv('D:/tensorflow learning/leaner_regression/hookelaw.csv')


# In[16]:


data


# In[17]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline 代码作用，当输入plt.plot(x,y_1)后，不必再输入 plt.show()，图像将自动显示出来


# In[18]:


plt.scatter(data.force, data.length_variation)
#散点图


# In[19]:


#求参数k和b，使得预测值和真实值之间的损失函数最小,均方差 (f(x)-y)²/n 最小
#使用梯度下降算法
x = data.force
y = data.length_variation


# In[75]:


#使用顺序模型 tf.keras.Sequential()
model = tf.keras.Sequential()


# In[76]:


#为模型添加层数layers 其中全连接层Dense层就是f(x) = kx + b 表示为output = activation(dot(input, kernel)+bias)
#全连接”表示上一层的每一个神经元，都和下一层的每一个神经元是相互连接的
#示例keras.layers.Dense(512, activation= 'sigmoid', input_dim= 2, use_bias= True)
#input_shape=(16,)等价于input_dim=16
#(1,)元组
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))   


# In[94]:


#显示模型概况
model.summary()  # ax + b


# In[97]:


# 因为有时候会误操作多加了几层，删除层数方法示例，调用.pop()
"""
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1
"""

model.compile(optimizer='adam',loss='mse')
#梯度下降算法常用'adam' , 均方差'mse'


# In[98]:


history = model.fit(x, y, epochs=5000)  #epochs对所有数据训练次数


# In[99]:


model.predict(x)


# In[102]:


model.predict(pd.Series([30]))  #数据类型是pd.Series，假设x为30，得出预测结果


# In[103]:


model.predict(pd.Series([15]))


# In[ ]:




