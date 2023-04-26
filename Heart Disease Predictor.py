#!/usr/bin/env python
# coding: utf-8

# ## Analysis packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#import file (excel)
df = pd.read_excel('C:/Users/herna/Documents/HERNÁN/Data Analysis/PGP Data Analytics/3_Courses (7)/7_Data Analyst Capstone/Project 1 - Healthcare/data.xlsx')


# ## EDA

# In[3]:


#head
df.head(5)


# In[4]:


#rows, columns
df.shape


# In[5]:


#unique values
df.nunique(axis=0)


# In[6]:


#nulls
df.isna().sum()


# In[7]:


#looking for duplicates
df.duplicated().sum()


# In[8]:


#delete duplicates
df = df.drop_duplicates()


# In[9]:


#rows, columns
df.shape


# In[10]:


#summary of count, min, max, mean, st dev
df.describe()


# In[11]:


#proportion of + and - binary predictor
df['target'].value_counts()


# ## Correlation Matrix

# In[12]:


#Basic heatmap
corr = df.corr()
sns.heatmap(corr)


# In[13]:


#Detailed heatmap
corr = df.corr()
plt.subplots(figsize = (15, 10))
sns.heatmap(corr,
            xticklabels = corr.columns,
            yticklabels = corr.columns,
            annot = True,
            cmap = sns.diverging_palette(220,20,as_cmap=True))


# ##### Respect to target
# ###### * positive correlation --> cp (+0.43), thalach (+0.42)
# ###### * negative correlation --> exang (-0.44), oldpeak (-0.43)

# ## Pairplots with continuous features

# In[14]:


sns.pairplot(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])


# ## ST depression (oldpeak) vs Heat disease (target)

# In[15]:


sns.catplot(x = 'target',
           y = 'oldpeak',
           hue = 'slope',
           kind = 'bar',
           data = df)


# ##### Same distribution for both heart conditions, low oldpeak means higher rish

# ## Box plot to check thalach vs target

# In[16]:


plt.figure(figsize = (10, 8))
sns.boxplot(x = 'target',
           y = 'thalach',
           hue = 'sex',
           data = df)


# ##### Similar distribution for male and females
# ##### Positive heart disease patients have higher median for thalach

# ## Filter positive (1) and negative (0) Heart Disease patients, to compare both oldpeak and thalach mean

# In[17]:


positive_df = df[df['target'] == 1]
positive_df.head()


# In[18]:


negative_df = df[df['target'] == 0]
negative_df.head()


# In[19]:


positive_df['oldpeak'].mean()


# In[20]:


negative_df['oldpeak'].mean()


# In[21]:


positive_df['thalach'].mean()


# In[22]:


negative_df['thalach'].mean()


# ##### Positive patients have 1/3 of the oldpeak respect the negative patients

# ## Model Building

# ### Logistic Regression

# In[23]:


# assign the 13 features to X and target to y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[24]:


# split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[25]:


# normalize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[26]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state = 1)    #get instance of model
model1.fit(X_train, y_train)                     #train/fit model

y_pred1 = model1.predict(X_test)                 #get y predictions
print(classification_report(y_test, y_pred1))    #accuracy


# ##### Logistic Regression Accuracy = 80%

# ### K-NN (K- Nearest Neighbors)

# In[27]:


from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier()                 #get instance of model
model2.fit(X_train, y_train)                    #train/fit model

y_pred2 = model2.predict(X_test)                #get y pred
print(classification_report(y_test, y_pred2))   #accuracy


# ##### K-NN Accuracy = 84%

# ### SVM (Support Vector Machine) 

# In[28]:


from sklearn.metrics import classification_report
from sklearn.svm import SVC

model3 = SVC(random_state = 1)                  #get instance of model
model3.fit(X_train, y_train)                    #train/fit model

y_pred3 = model3.predict(X_test)                #get y pred
print(classification_report(y_test, y_pred3))   #accuracy


# ##### SVM Accuracy = 79%

# ### Naives Bayes Classifier

# In[29]:


from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB()                           #get instance of model
model4.fit(X_train, y_train)                    #train/fit model

y_pred4 = model4.predict(X_test)                #get y pred
print(classification_report(y_test, y_pred4))   #accuracy


# ##### Naives Bayes Classifier Accuracy = 82%

# ### Decision Trees

# In[30]:


from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state = 1)         #get instance of model
model5.fit(X_train, y_train)                              #train/fit model

y_pred5 = model5.predict(X_test)                          #get y pred
print(classification_report(y_test, y_pred5))             #accuracy


# ##### Decision Trees Accuracy = 75%

# ### Random Forest

# In[31]:


from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state = 1)         #get instance of model
model6.fit(X_train, y_train)                              #train/fit model

y_pred6 = model6.predict(X_test)                          #get y pred
print(classification_report(y_test, y_pred6))             #accuracy


# ##### Random Forest Accuracy = 80%

# #### Highest Accuracy for K-NN (84%)

# ## Confusion Matrix

# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred2)      #y_pred2 for K-NN
print(cm)
accuracy_score(y_test, y_pred2)


# ###### TP = 23, FP = 6
# ###### FN = 4, TN = 28
# ###### Accuracy = 83.61%

# ## Predictions

# ### Scenario
# ##### age = 20
# ##### sex = 1
# ##### cp =  2
# ##### trestbps =  110
# ##### chol = 230
# ##### fbs > 120 --> 1
# ##### restecg =  
# ##### thalach =  
# ##### exang =  1
# ##### oldpeak = 2.2  
# ##### slope =  2
# ##### ca = 0
# ##### thal = 2

# In[33]:


print(model2.predict(sc.transform([[20, 1, 2, 110, 230, 1, 1, 140, 1, 2.2, 2, 0, 2]])))


# #### Binary output = 1, therefore Diagnosis of Heart Disease

# ## Conclusions

# ### - Cp, oldpeak and thalach are the most significant features that help to classify between a positive or negative diagnosis
# ### - The model is ready to classify patients 
# ### - The K-NN model has over 80% accuracy, which is above 70%, so it is ideal 
