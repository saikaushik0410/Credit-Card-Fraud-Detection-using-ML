#!/usr/bin/env python
# coding: utf-8

# In[40]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
import sklearn


# In[41]:


# loading the dataset using pandas 
data = pd.read_csv('creditcard.csv')


# In[42]:


a = np.array([[1,2,3],[0,0,0]])
print(a)


# In[43]:


print(data.columns)


# In[44]:


print(data.shape)


# In[45]:


print(data.describe())


# In[46]:


data = data.sample(frac = 1, random_state = 1)
print(data.shape)


# In[47]:


# plot a histogram 
data.hist(figsize = (20,20))
plt.show()


# In[48]:


# determining the number of fraud cases

fraud = data[data['Class']==1]
valid = data[data['Class']==0]

print('Fraud cases: {}'.format(len(fraud)))
print('Valid cases: {}'.format(len(valid)))


# In[49]:


#The Fraud transactions
print(fraud)


# In[19]:


#the Valid cases
print(valid)


# In[50]:


# to determine the outliers in the data 
outlier = len(fraud) / float(len(valid))
print(outlier)


# In[51]:


# Correlation matrix - to check for correlations
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[52]:


#To get the columns from dataframe
coloumns = data.columns.tolist()

#to filter data
columns = [c for c in coloumns if c not in ["Class"]]

#Store the predicting variable 
target = "Class"

x = data[columns]
y = data[target]

print(x.shape)
print(y.shape)


# In[53]:


#Isolation forest algorithm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#defining a random state
state= 1

#Outlier detection 
classifiers = {"Isolation forest": IsolationForest(max_samples=len(x), contamination=outlier, random_state=state),
               "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20 , contamination=outlier, novelty=True)}


# In[54]:


#Fitting the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    #fitting the data and tag outliers
    if clf_name == "Local Outlier Factor ":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred  = clf.predict(x)
        
    #Reshaping the predictions values to 0 - valid and 1 - Fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    #classification metrics
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))


# In[ ]:




