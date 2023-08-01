#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("pistachio.csv")
df


# In[3]:


df['Class'].value_counts()


# In[4]:


df.isnull().sum()


# In[5]:


#FUNCTION OF LABEL ENCODER
#converts categorical column into numerical data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Class']=le.fit_transform(df['Class'])


# In[6]:


df


# In[7]:


correlation_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='rocket_r', linewidths=0.5, fmt='.2f')


# In[8]:


sns.scatterplot(x='ROUNDNESS', y='COMPACTNESS', hue='Class', data=df)


# In[6]:


X = df.drop(columns=[ 'Class'],axis=1)
y = df['Class']


# In[ ]:


#DECISION TREES
#IT IS A SUPERVISED MACHINE LEARNING ALGORITHM WHICH IS USED FOR CLASSIFICATION PROBLEM.IT IS A TREE STRUCTURED CLASSIFIER 
#WHERE INTERNAL NODES REPRESENT FEATURES OF A DATASET, EACH NODES REPRESENT DECISION RULE.


# In[ ]:





# In[8]:


#standardization is a scaling technique where it makes the data scale free by converting the data into mean 0 and sd 1.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[5]:


hyperparameters = [{
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10]
}]

dt = DecisionTreeClassifier()
grid_search = GridSearchCV(dt, hyperparameters, scoring='f1', cv=5, verbose=True, n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[20]:


best_dt_params=grid_search.best_params_
best_dt_params


# In[19]:


best_dt = DecisionTreeClassifier(**best_dt_params)
best_dt.fit(X_train, y_train.ravel())
y_pred_dt = best_dt.predict(X_test)
f1 = f1_score(y_test, y_pred_dt)
print("F1 Score on Test Set:", f1)


# In[21]:


dt_cm = confusion_matrix(y_test, y_pred_dt)
sns.set(font_scale=1.5)
plt.figure(dpi=70)
sns.heatmap(dt_cm, annot=True, cmap='Greens', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[22]:


print("Accuracy:", metrics.accuracy_score(y_test,y_pred_dt))


# In[23]:


hyperparameters = [{
    'n_estimators': range(10,120,10),
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 5, 10],
    'max_features':[2,3,4]
}]

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, hyperparameters, scoring='f1', cv=5, verbose=True, n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[25]:


best_rf_params = grid_search.best_params_
print(best_rf_params)


# In[26]:


best_rf = RandomForestClassifier(**best_rf_params)
best_rf.fit(X_train, y_train.ravel())
y_pred_rf = best_rf.predict(X_test)
f1 = f1_score(y_test, y_pred_rf)
print("F1 Score on Test Set:", f1)


# In[27]:


print("Accuracy:", metrics.accuracy_score(y_test,y_pred_rf))


# In[28]:


rf_cm = confusion_matrix(y_test, y_pred_rf)
sns.set(font_scale=1.5)
plt.figure(dpi=70)
sns.heatmap(rf_cm, annot=True, cmap='Greens', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[29]:


lr_params = [{'C': [0.1, 1, 10]}]

lr = LogisticRegression()
lr_grid_search = GridSearchCV(lr, lr_params, scoring='f1', cv=5, verbose=True, n_jobs=-1)
lr_grid_search.fit(X_train, y_train)


# In[32]:


best_lr_params = lr_grid_search.best_params_
print(best_lr_params)
best_lr = LogisticRegression(**best_lr_params)
best_lr.fit(X_train, y_train)
y_pred_lr = best_lr.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr)
print("F1 Score on Test Set (Logistic Regression):", f1_lr)


# In[33]:


print("Accuracy (Logistic Regression):", metrics.accuracy_score(y_test, y_pred_lr))


# In[34]:


lr_cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(dpi=70)
sns.heatmap(lr_cm, annot=True, cmap='Greens', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# In[ ]:




