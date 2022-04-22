#!/usr/bin/env python
# coding: utf-8

# ## Importing Independencies

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## Data Collection and Data Processing

# In[3]:


# Loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv("D:\AI\Machine Learning\Projects\Sonar Rock vs Mine Prediction\Sonar Data.csv", header = None)


# In[13]:


sonar_data.head()


# In[14]:


# The number of rows and columns 
sonar_data.shape


# In[15]:


# Let's try to get some statistical definitions of this data
sonar_data.describe() # Describe --> Statistical measures of the data


# In[17]:


sonar_data[60].value_counts()


# M --> Mine
# 
# R --> Rock

# In[19]:


# Separating data and labels
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60] # Storing last column in Y


# In[20]:


print(X)
print(Y)


# ## Training and test data 

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# In[29]:


print(X_train)
print(Y_train)


# ## Model training --> Logistic Regression

# In[23]:


model = LogisticRegression()


# In[31]:


# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# ## Model Evaluation

# In[32]:


# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data: ", training_data_accuracy)


# In[35]:


# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data: ', test_data_accuracy)


# ## Making a predictive system

# In[55]:


# input_data = (0.0094,0.0166,0.0398,0.0359,0.0681,0.0706,0.1020,0.0893,0.0381,0.1328,0.1303,0.0273,0.0644,
#               0.0712,0.1204,0.0717,0.1224,0.2349,0.3684,0.3918,0.4925,0.8793,0.9606,0.8786,0.6905,0.6937,
#               0.5674,0.6540,0.7802,0.7575,0.5836,0.6316,0.8108,0.9039,0.8647,0.6695,0.4027,0.2370,0.2685,
#               0.3662,0.3267,0.2200,0.2996,0.2205,0.1163,0.0635,0.0465,0.0422,0.0174,0.0172,0.0134,0.0141,
#               0.0191,0.0145,0.0065,0.0129,0.0217,0.0087,0.0077,0.0122)
input_data = (0.0363,0.0478,0.0298,0.0210,0.1409,0.1916,0.1349,0.1613,0.1703,0.1444,0.1989,0.2154,0.2863
              ,0.3570,0.3980,0.4359,0.5334,0.6304,0.6995,0.7435,0.8379,0.8641,0.9014,0.9432,0.9536,1.0000
              ,0.9547,0.9745,0.8962,0.7196,0.5462,0.3156,0.2525,0.1969,0.2189,0.1533,0.0711,0.1498,0.1755
              ,0.2276,0.1322,0.1056,0.1973,0.1692,0.1881,0.1177,0.0779,0.0495,0.0492,0.0194,0.0250,0.0115
              ,0.0190,0.0055,0.0096,0.0050,0.0066,0.0114,0.0073,0.0033)

# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')


# In[ ]:




