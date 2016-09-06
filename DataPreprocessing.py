
# coding: utf-8

# In[29]:

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import auc, f1_score, log_loss
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam, Adagrad
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical


# In[58]:

train_dat = pd.read_csv('data/train.csv')
labelEncoder = LabelEncoder()
onehotEncoder = OneHotEncoder()
train_dat.head()
new_dat = np.log(train_dat.drop(['id','target'],axis=1)+1)
kfold = StratifiedKFold(labelEncoder.fit_transform(train_dat.target[:X_train.shape[0]]), n_folds=10, shuffle=True)
categories = labelEncoder.fit_transform(train_dat.target).reshape(-1,1)
onehot_categorical = onehotEncoder.fit_transform(categories.reshape(-1,1)).toarray()
dummy_y = to_categorical(categories)
X_train,X_test,y_train, y_test = train_test_split(new_dat,onehot_categorical,test_size=0.3,stratify=categories)


# In[71]:

hidden_nodes = 10
output_node = labelEncoder.classes_.shape[0]
model = Sequential([
    Dense(hidden_nodes,input_dim=X_train.shape[1], activation='sigmoid',init='uniform'),
    Dense(output_node, init='uniform'),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[60]:

def create_model():
    hidden_nodes = 10
    output_node = labelEncoder.classes_.shape[0]
    model = Sequential([
        Dense(hidden_nodes,input_dim=X_train.shape[1], activation='sigmoid',init='uniform'),
        Dense(output_node, init='uniform'),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[61]:

model.summary()


# In[65]:

sk_model = KerasClassifier(build_fn=create_model,nb_epoch=10, batch_size = 10)


# In[66]:

scores = cross_val_score(sk_model, X_train.values,y_train, cv=kfold, n_jobs=1, scoring='log_loss')


# In[69]:

scores.mean()


# In[70]:

# history = model.fit(X_train.values,y_train, nb_epoch=200, batch_size=20, verbose=2, validation_data=(X_test.values,y_test))


# In[25]:

y_pred = model.predict_proba(X_test.values, batch_size=10)


# In[26]:

loss_model= log_loss(y_test, y_pred)


# In[ ]:

test = pd.read_csv('data/test.csv')
new_test = test.drop(['id'],axis=1)
y_test = model.predict_proba(new_test.values)


# In[108]:

submisstion = pd.read_csv('data/sampleSubmission.csv')
nn_sub = pd.DataFrame(columns=submisstion.columns, index=test.index)
nn_sub.id = test.id
nn_sub[nn_sub.columns[1:]] = y_test
nn_sub.to_csv("nn_hidden_10_%s.csv"%loss_model,index=None)

