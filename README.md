# imbalancedlearning_Over sampling

# setting
import numpy as np
import scipy 
import pandas as pd
import sklearn
import imblearn
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

y = data_set.iloc[:,0:1]
x = data_set.iloc[:,1:]


############
## 1.Smote #
############
sm = SMOTE(ratio=0.12)#10%까지만 증가
x_sm_res , y_sm_res = sm.fit_sample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_sm_res,y_sm_res,test_size=0.3,random_state=1234)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

#파라미터
param = {
   'objective': 'binary:logistic',
    'silent': 1,
    'eta': 0.1
}
num_round = 300

#fit
model = xgb.train(param,dtrain,num_round)
y_pred = model.predict(dtest)
preds = [1 if (value >= 0.5) else 0 for value in y_pred]
sklearn.metrics.f1_score(y_test,preds)

########################
# 2. Borderline-smote  #
########################
bd = SMOTE(ratio=0.12,kind='borderline1')#10%까지만 증가
x_bd_res , y_bd_res = bd.fit_sample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_bd_res,y_bd_res,test_size=0.3,random_state=1234)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

#파라미터
param = {
   'objective': 'binary:logistic',
    'silent': 1,
    'eta': 0.1
}
num_round = 300

#fit
model = xgb.train(param,dtrain,num_round)
y_pred = model.predict(dtest)
preds = [1 if (value >= 0.5) else 0 for value in y_pred]
sklearn.metrics.f1_score(y_test,preds)


#############
# 3. ADASYN #
#############

ada = ADASYN(ratio=0.12)
x_ada_res , y_ada_res = ada.fit_sample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_ada_res,y_ada_res,test_size=0.3,random_state=1234)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

#파라미터
param = {
   'objective': 'binary:logistic',
    'silent': 1,
    'eta': 0.1
}
num_round = 300

#fit
model = xgb.train(param,dtrain,num_round)
y_pred = model.predict(dtest)
preds = [1 if (value >= 0.5) else 0 for value in y_pred]
sklearn.metrics.f1_score(y_test,preds)



