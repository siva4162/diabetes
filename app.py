# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection  import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import pickle

data=pd.read_csv("pima-data.csv")
#data=df.drop(['skin'],axis=1)
corrmat=data.corr()
top_corr_features= corrmat.index
#print(corrmat)
diabetes_map={True:1,False:0}
data['diabetes']=data['diabetes'].map(diabetes_map)
#print(data.diabetes)
#print(data['diabetes'].value_counts())

X=data.drop(['diabetes',],axis=1)
y=data.diabetes
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
# missing values 

fill_valus=SimpleImputer(missing_values= 0,strategy='mean',verbose=0)
X_train=fill_valus.fit_transform(X_train)
X_test=fill_valus.fit_transform(X_test)

re=RandomForestClassifier(random_state=10)
re.fit(X_train,y_train.ravel())
print(re.score(X_test,y_test))
param={
       "learning_rate" :[0.05,0.10,0.15,0.20,0.25,0.30],
       "max_depth":[3,4,5,6,8,10,12,15],
       "min_child_weight":[1,3,5,7],
       "gamma":[0.0,0.1,0.2,0.3,0.4,0.]}
classfier= xgboost.XGBClassifier()
s=RandomizedSearchCV(classfier, param_distributions=param,verbose=3,n_jobs=-1)
s.fit(X_train,y_train.ravel())
#print(s.best_estimator_)
k=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0.0,
              learning_rate=0.05, max_delta_step=0, max_depth=10,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


k.fit(X_train,y_train.ravel())
s=k.predict(X_test)
y=k.predict([[2,141,58,	34,128,25.4,0.699,24,1.3396]])
pickle.dump(k,open("sugar.pkl",'wb'))
model=pickle.load(open("sugar.pkl",'rb'))
print(data.columns)
print(data.shape)


