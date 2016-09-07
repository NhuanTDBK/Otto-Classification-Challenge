
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import pickle
from utils import *


# In[3]:

n_iter = 100
k_fold = 10
# cv = kfold
# initialize the classifier
X_train, X_val, y_train, y_val, cv = load_train_and_kfold(n_folds=k_fold)


# In[13]:

model = GradientBoostingClassifier(random_state=4111)
model_name = model.__class__.__name__
param_grid = {
     'learning_rate': sp_uniform(loc=0e0,scale=1e0),
      "max_depth": sp_randint(4, 100),
      "max_features": sp_randint(1, 11),
      "min_samples_split": sp_randint(1, 11),
      "min_samples_leaf": sp_randint(1, 11),
      'subsample': np.arange(0.6,1.0,step=0.05),
      "n_estimators": sp_randint(100,600)
}


# In[ ]:

search_GB = RandomizedSearchCV(model,param_grid,scoring='log_loss',n_jobs=-1,
               n_iter=n_iter,cv=cv,verbose=True)
search_GB.fit(X_train,y_train.flatten())


# In[ ]:

log_model = search_GB.score(X_val,y_val.flatten())
print "Log loss = %s"%log_model
X_test = get_test()
y_pred = search_GB.predict_proba(X_test)
save_submission(model_name,log_model,y_pred)

