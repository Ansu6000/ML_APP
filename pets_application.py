#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()


# In[4]:


im = PILImage.create('dog.jpg')
im.thumbnail((192,192))
im


# In[5]:


learn = load_learner('model1.pkl')


# In[6]:


learn.predict(im)


# In[7]:


categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))


# In[9]:


classify_image(im)


# In[12]:


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False, share=True)


# In[ ]:




