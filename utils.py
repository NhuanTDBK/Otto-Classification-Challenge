import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_onehot_train_and_kfold(n_folds=10):
	train_dat = pd.read_csv('data/train.csv')
	labelEncoder = LabelEncoder()
	onehotEncoder = OneHotEncoder()
	new_dat = np.log(train_dat.drop(['id','target'],axis=1)+1)
	categories = labelEncoder.fit_transform(train_dat.target).reshape(-1,1)
	onehot_categorical = onehotEncoder.fit_transform(categories.reshape(-1,1)).toarray()
	X_train,X_test,y_train, y_test = train_test_split(new_dat,onehot_categorical,test_size=0.3,stratify=categories)
	kfold = StratifiedKFold(labelEncoder.fit_transform(train_dat.target[:X_train.shape[0]]), n_folds=n_folds, shuffle=True)
	return X_train,X_test,y_train, y_test, kfold
def load_train_and_kfold(n_folds=10):
	train_dat = pd.read_csv('data/train.csv')
	labelEncoder = LabelEncoder()
	onehotEncoder = OneHotEncoder()
	train_dat.head()
	new_dat = train_dat.drop(['id','target'],axis=1)
	categories = labelEncoder.fit_transform(train_dat.target).reshape(-1,1)
	X_train,X_test,y_train, y_test = train_test_split(new_dat,categories,test_size=0.3,stratify=categories)
	kfold = StratifiedKFold(labelEncoder.fit_transform(train_dat.target[:X_train.shape[0]]), n_folds=n_folds, shuffle=True)
	return X_train,X_test,y_train, y_test, kfold 

def save_submission(model_name,loss_model,y_test):
	test = pd.read_csv('data/test.csv')
	new_test = test.drop(['id'],axis=1)
	#y_test = model.predict_proba(new_test.values)
	submisstion = pd.read_csv('data/sampleSubmission.csv')
	nn_sub = pd.DataFrame(columns=submisstion.columns, index=test.index)
	nn_sub.id = test.id
	nn_sub[nn_sub.columns[1:]] = y_test
	log_loss = loss_model
	nn_sub.to_csv("%s_%s.csv"%(model_name,log_loss),index=None)
	print "Save submission completed"
def load_train(file_name="train_data"):
	dat_file = np.load(file_name)
	return dat_file["X_train"],dat_file["X_test"],dat_file["y_train"],dat_file["y_test"]
def get_test(normalize=True):
	test = pd.read_csv('data/test.csv')
	new_test = test.drop(['id'],axis=1)
	if normalize is True:
		return np.log(new_test+1)
	return new_test
