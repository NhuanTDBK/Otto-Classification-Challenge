
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import pickle
from utils import *


# In[2]:

n_iter = 50
k_fold = 3
# cv = kfold
# initialize the classifier


# In[3]:
X_train, X_val, y_train, y_val, cv = load_train_and_kfold(n_folds=k_fold)

# In[4]:

GB = xgb.XGBClassifier(silent=False)
param_grid = {
              'max_depth': sp_randint(4, 200),
              'learning_rate': sp_uniform(loc=0e0,scale=1e0),
              'objective':['multi:softprob'],
              'nthread': [8],
              'missing': [np.nan],
              'reg_alpha': [0.01,0.017782794,0.031622777,0.056234133,\
                            0.1,0.17782794,0.31622777,0.56234133,1.,1.77827941,\
                            3.16227766,5.62341325,10.,\
                            17.7827941,31.6227766,56.2341325,100.],
              'colsample_bytree': sp_uniform(loc=0.2e0,scale=0.8e0),
              'subsample': np.arange(0.6,1.0,step=0.05),
              'n_estimators': sp_randint(100,700),
}

print "Randomized XGBoost"
# In[ ]:
for i in range(2):
	print "Loop %i/20"%i
	search_GB = RandomizedSearchCV(GB,param_grid,scoring='log_loss',n_jobs=-1,
               n_iter=n_iter,cv=cv,verbose=True)
	search_GB.fit(X_train,y_train)
	log_model = search_GB.score(X_val,y_val)
	print "Log loss = %s"%log_model
	X_test = get_test()
	save_submission('XGBoost',log_model,search_GB.predict_proba(X_test))
