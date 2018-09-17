#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import dtree



# In[2]:


# create a decision tree object
dt = dtree.dtree()


# In[3]:


fileName = raw_input('Enter file name: ')


# In[4]:


fh = open(fileName)
# load your data in for building the tree

data, classData, featureNames = dt.read_data(fileName)
print 'Feature Data: ', data
print 'Class Data:', classData
print 'feature Names: ', featureNames


# In[5]:





# build the decision tree model
t = dt.ID3(data,classData,featureNames, "")
print "Tree stored as a dictionary: ", t

#print out the decision tree model
print "\n-------------------"
print "Decision Tree Model:"
print "-------------------\n"
dt.printTree(t, "")

predicted = dt.classifyAll(t, data)
print "\nPreidction Accuracy: ", dt.predictionAccuracy(predicted, classData)


# In[ ]:




