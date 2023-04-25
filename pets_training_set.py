#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *


# In[2]:


path = untar_data(URLs.PETS)/'images'


# In[3]:


def is_cat(x): return x[0].isupper()


# In[4]:


dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))


# In[6]:


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


# In[7]:


learn.export('model1.pkl')


# In[ ]:




