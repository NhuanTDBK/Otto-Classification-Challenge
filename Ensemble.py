
# coding: utf-8

# In[34]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import auc, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# In[14]:

train_dat = pd.read_csv('data/train.csv')
labelEncoder = LabelEncoder()
onehotEncoder = OneHotEncoder()
train_dat.head()
new_dat = train_dat.drop(['id','target'],axis=1)
categories = labelEncoder.fit_transform(train_dat.target).reshape(-1,1)
onehot_categorical = onehotEncoder.fit_transform(categories.reshape(-1,1)).toarray()


# In[15]:

X_train,X_test,y_train, y_test = train_test_split(new_dat,categories,test_size=0.3,stratify=categories)


# In[16]:

clfs = []


# In[ ]:

rfc = RandomForestClassifier(n_estimators=500, random_state=4111, n_jobs=-1)
rfc.fit(X_train,y_train)
print('RFC LogLoss {score}'.format(score=log_loss(y_test, rfc.predict_proba(X_test))))
clfs.append(rfc)


# In[23]:

logreg = LogisticRegression(random_state=4111)
logreg.fit(X_train, y_train)
print('LogisticRegression LogLoss {score}'.format(score=log_loss(y_test, logreg.predict_proba(X_test))))
clfs.append(logreg)


# In[25]:

predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(X_test))


# In[33]:

starting_values = [0.5]*len(predictions)


# In[40]:

params = {"objective": "multi:softprob", "num_class": 9}
xgb_model = xgb.XGBClassifier(objective='multi:softprob')


# In[41]:

xgb_model.fit(X_train.values,y_train)


# In[43]:


print log_loss(y_test,xgb_model.predict_proba(X_test.values))

