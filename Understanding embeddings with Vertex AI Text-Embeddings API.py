#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install google-cloud-aiplatform')


# In[2]:


import vertexai
vertexai.init()


# In[3]:


from vertexai.language_models import TextEmbeddingModel
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")


# In[4]:


embedding = embedding_model.get_embeddings(['life'])


# In[5]:


vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])


# In[6]:


embedding = embedding_model.get_embeddings(['What is the meaning of life?'])


# In[7]:


vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])


# In[8]:


from sklearn.metrics.pairwise import cosine_similarity


# In[9]:


emb_1 = embedding_model.get_embeddings(['What is the meaning of life?'])
emb_2 = embedding_model.get_embeddings(['How does one spend their time well on Earth?'])
emb_3 = embedding_model.get_embeddings(['Would you like a salad?'])

vec_1 = [emb_1[0].values]
vec_2 = [emb_2[0].values]
vec_3 = [emb_3[0].values]


# In[10]:


print(cosine_similarity(vec_1,vec_2)) 
print(cosine_similarity(vec_2,vec_3))
print(cosine_similarity(vec_1,vec_3))


# In[11]:


in_1 = "Missing flamingo discovered at swimming pool"
in_2 = "Sea otter spotted on surfboard by beach"
in_3 = "Baby panda enjoys boat ride"
in_4 = "Breakfast themed food truck beloved by all!"
in_5 = "New curry restaurant aims to please!"
in_6 = "Python developers are wonderful people"
in_7 = "TypeScript, C++ or Java? All are great!" 

input_text_lst_news = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]


# In[12]:


embeddings = []
for input_text in input_text_lst_news:
    emb = embedding_model.get_embeddings(
        [input_text])[0].values
    embeddings.append(emb)


# In[13]:


import numpy as np
embeddings_array = np.array(embeddings) 
print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)


# In[14]:


from sklearn.decomposition import PCA

# Perform PCA for 2D visualization
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings_array)
new_values = PCA_model.transform(embeddings_array)


# In[15]:


print("Shape: " + str(new_values.shape))
print(new_values)


# In[16]:


get_ipython().system('pip install ipympl plot-utils matplotlib seaborn')


# In[17]:


import seaborn as sns
import pandas as pd

data = pd.DataFrame({ 'x':new_values[:,0], 'y':new_values[:,1], 'sentences': input_text_lst_news})

# Create a visualization
sns.relplot(
    data,
    x='x',
    y='y',
    kind='scatter',
    hue='sentences'
)


# In[ ]:




