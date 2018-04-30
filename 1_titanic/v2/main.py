#!/usr/bin/env python
# v0.2
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#0) Preprocessing
train = train.fillna(0)	#FIXME
test = test.fillna(0)	#FIXME

train['Sex'] = train['Sex'].replace('male',0)
train['Sex'] = train['Sex'].replace('female',1)
test['Sex'] = test['Sex'].replace('male',0)
test['Sex'] = test['Sex'].replace('female',1)
print train[0:5]

r = smf.ols('Survived ~ Age + Sex + Pclass + SibSp',data=train).fit()
print r.summary()
0/0

reg = LogisticRegression()
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for kftrain, kftest in kf.split(train):
	


0/0
reg.fit( train[['Age','Pclass','Sex','SibSp']], train['Survived'] )

results = reg.predict( test[['Age','Pclass','Sex','SibSp']] )
ids = np.asarray( test['PassengerId'] )
final_result = pd.DataFrame( {'PassengerId':ids, 'Survived':results})
print final_result.to_csv(index=False)
#df = pd.DataFrame( test[['PassengerId']], final_results )
#result_real = test[['Survived']]

