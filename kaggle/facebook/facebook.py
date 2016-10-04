

import csv as csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm


# # Read Data 

# In[2]:

df_train = pd.read_csv('./train/train.csv')
df_train.reindex(np.random.permutation(df_train.index))
df_test = pd.read_csv('./test/test.csv')


# In[65]:

print(df_train.describe())


# ## Split Data 

# In[8]:

x = df_train[df_train.columns[1:5]]
y = df_train['place_id']




# # Classification based on KClusters

# In[82]:

num_clusters = 10000
clusterIdx = y.value_counts()[:num_clusters].keys().get_values()


# In[63]:

print(clusterIdx)


# In[ ]:

clusters = []

for cluster in clusterIdx:
    temp = [0, 0, 0, 0]
    tempNum = 0
    for row in df_train.get_values():
        if row[5] == cluster:
            temp[0] += row[1]
            temp[1] += row[2]
            temp[2] += row[3]
            temp[3] += row[4]
    temp[0] /= tempNum
    temp[1] /= tempNum
    temp[2] /= tempNum
    temp[3] /= tempNum
    clusters.append(temp)
