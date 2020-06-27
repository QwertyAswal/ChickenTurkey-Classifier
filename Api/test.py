#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


input_ = tf.keras.layers.Input(shape=[224,224,3])
conv1 = tf.keras.layers.Conv2D(64, 5, activation= 'relu', padding='same')(input_)
pool1 = tf.keras.layers.MaxPool2D(2)(conv1)
conv2 = tf.keras.layers.Conv2D(128, 5, activation= 'relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPool2D(2)(conv2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation= 'relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPool2D(2)(conv3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation= 'relu', padding='same')(pool3)
pool4 = tf.keras.layers.MaxPool2D(2)(conv4)
conv5 = tf.keras.layers.Conv2D(1024, 3, activation= 'relu', padding='same')(pool4)
pool5 = tf.keras.layers.MaxPool2D(2)(conv5)
conv6 = tf.keras.layers.Conv2D(2048, 3, activation= 'relu', padding='same')(pool5)
pool6 = tf.keras.layers.MaxPool2D(2)(conv6)
flat = tf.keras.layers.Flatten()(pool6)
dense1 = tf.keras.layers.Dense(1024, activation= 'relu')(flat)
drop1 = tf.keras.layers.Dropout(0.5)(dense1)
dense2 = tf.keras.layers.Dense(256, activation= 'relu')(drop1)
drop2 = tf.keras.layers.Dropout(0.5)(dense2)
dense3 = tf.keras.layers.Dense(64, activation= 'relu')(drop2)
dense4 = tf.keras.layers.Dense(16, activation= 'relu')(dense3)
output_ = tf.keras.layers.Dense(2, activation= 'softmax')(dense4)


# In[3]:


model = tf.keras.Model(inputs=[input_], outputs=[output_])


# In[4]:


model.load_weights('withoutResNetWeights.h5')


# In[5]:


import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import pathlib


# In[7]:


img = Image.open('test/turkey1.jpg')


# In[8]:


img


# In[15]:


te = np.asarray(img.resize((224,224)))


# In[26]:


te = [te/255]


# In[28]:


test = np.array(te)


# In[29]:


test.shape


# In[31]:


print(model.predict(test))


# In[25]:





# In[ ]:




