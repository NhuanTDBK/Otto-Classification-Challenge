
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import auc, f1_score, log_loss
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical


# In[19]:

train_dat = pd.read_csv('data/train.csv')
labelEncoder = LabelEncoder()
onehotEncoder = OneHotEncoder()
train_dat.head()
new_dat = np.log(train_dat.drop(['id','target'],axis=1)+1)
categories = labelEncoder.fit_transform(train_dat.target).reshape(-1,1)
onehot_categorical = onehotEncoder.fit_transform(categories.reshape(-1,1)).toarray()
dummy_y = to_categorical(categories)
X_train,X_test,y_train, y_test = train_test_split(new_dat,onehot_categorical,test_size=0.3,stratify=categories)
np.savez("train_data",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# In[20]:
batch_size = 64
nb_epoch = 512
hidden_nodes = 2048
output_node = labelEncoder.classes_.shape[0]
model = Sequential([
    Dense(hidden_nodes,input_dim=X_train.shape[0],activation='sigmoid',init='uniform'),
    Dropout(0.5),	
    Dense(4096,init='uniform',activation='sigmoid'),
    Dropout(0.16),
    Dense(output_node, init='uniform'),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# In[21]:

model.summary()
# In[ ]:

history = model.fit(X_train.values,y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2,validation_split=0.1)


# In[25]:

y_pred = model.predict_proba(X_test.values, batch_size=batch_size)


# In[26]:

loss_model= log_loss(y_test, y_pred)


# In[ ]:

test = pd.read_csv('data/test.csv')
new_test = np.log(test.drop(['id'],axis=1)+1)
y_test = model.predict_proba(new_test.values)

print "Log loss = %s"%loss_model
# In[108]:

submisstion = pd.read_csv('data/sampleSubmission.csv')
nn_sub = pd.DataFrame(columns=submisstion.columns, index=test.index)
nn_sub.id = test.id
nn_sub[nn_sub.columns[1:]] = y_test
nn_sub.to_csv("nn_hidden_10_%s.csv"%loss_model,index=None)

