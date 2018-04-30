#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


#################
# 0) Parameters
################

grid_search=False

train=pd.read_csv('train.csv') 
len_train=len(train)
test=pd.read_csv('test.csv')
data=pd.concat([train,test],ignore_index=True)
target='SalePrice'
qexp_0=['OverallQual','GrLivArea','TotalBsmtSF','1stFlrSF','YearBuilt',\
	'TotRmsAbvGrd','YearRemodAdd','BsmtFinSF1','MasVnrArea','2ndFlrSF',\
	'LotArea','LotFrontage']
# category to number		[good, ok, bad] -> [2, 1, 0]
qexp_1=['ExterQual','BsmtQual','BsmtFinType1','BsmtExposure']
# dummies		[blue, yellow, pink] -> blue=1/0, yellow=1/0, ...
qexp_2=['Neighborhood','MSSubClass']

qexp=qexp_0+qexp_1+qexp_2

###################
#1 Prepare data
###################
all_vars=[u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
u'Street', u'Alley', u'LotShape', u'LandContour', u'Utilities',
u'LotConfig', u'LandSlope', u'Neighborhood', u'Condition1',
u'Condition2', u'BldgType', u'HouseStyle', u'OverallQual',
u'Fence', u'MiscFeature', u'MiscVal', u'MoSold', u'YrSold', u'SaleType',
u'SaleCondition']

for v in all_vars:
	if v not in qexp:
		data=data.drop(v,axis=1)


### category encoding
from sklearn.preprocessing import LabelEncoder
encs=[]
for v in qexp_1:
	le = LabelEncoder()
	data[v] = data[v].fillna('NAN')
	le.fit(data[v].unique())
	data[v] = le.transform(data[v])
	encs.append(le)

### dummy adding
for v in qexp_2:
	data = pd.get_dummies(data,columns=[v])
#	data=data.drop('Neighborhood',axis=1)

s="""
print data['PoolQC']
data2=data['PoolQC']
cols_to_transform = [ 'PoolQC' ]
data2 = pd.get_dummies(data2, columns=cols_to_transform )	#dummy trans
print data2.head()
print data2[qexp].head()
0/0"""

data = data.fillna(0)
test_id=data[len(train):(len(train)+len(test))]['Id']
data=data.drop('Id',axis=1)
train_x=data[0:len(train)].drop(target,axis=1)
train_y= data[0:len(train)][target]
test_x=data[len(train):(len(train)+len(test))].drop(target,axis=1)
###################
#2 Analyse data
###################

##################
#3 Model data
###################

#import sys
#sys.path.insert(0, '../helper')
#from meta_predictor import BestRegressor

#meta = BestRegressor(train_x, train_y, 4, 'r2', 0)
#meta.evaluate()
#0/0

import seaborn as sns
import matplotlib.pyplot as plt
#sns.distplot(train_y)
#plt.show()
#train_y=np.log1p(train_y)

#test_y=np.log1p(test_y)
#sns.distplot(train_y)
#plt.show()

#0/0
from sklearn.svm import LinearSVR
svm = LinearSVR(C=1.2,random_state=0,loss='epsilon_insensitive')
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

regs=[]
regs.append( LinearRegression() )
regs.append( Ridge(alpha=1.5))# penalizing value of coefficients (Wert)
regs.append( Lasso(alpha=0.5))# penalizing number of coefficients (Anzahl)
regs.append( svm )
regs.append( AdaBoostRegressor(n_estimators=120,learning_rate=0.2) )
regs.append( RandomForestRegressor(n_estimators=50,max_depth=8) )
regs.append( GradientBoostingRegressor(n_estimators=150,max_depth=3) )
regs.append( XGBRegressor(n_estimators=275,max_depth=3,\
						early_stopping_rounds=5) )
infos=[]
for r in regs:
	infos.append( r.__class__.__name__ )

qmetric='r2'
for reg,info in zip(regs,infos):
	scores=cross_val_score(reg, train_x, train_y, cv=7, \
			scoring=qmetric)
	print("%s: %0.3f (+/- %0.2f) [%s]"%(qmetric,scores.mean(),scores.std(),\
			info))


##################
#5 Predict data
###################
if True:
	
	# single model or complex model
	############
	if False:
		reg=GradientBoostingRegressor(n_estimators=100,max_depth=4)
		reg.fit(train_x,train_y)
		results = reg.predict( test_x ).astype(int)

	else:
		reg=LinearRegression()
		tmp_df=pd.DataFrame()
		final_reg=[regs[6],regs[5],regs[4]]
		for f in final_reg:
			f.fit(train_x,train_y)
			tmp_df['%s_pred'%(f.__class__.__name__)]=f.predict(\
					pd.concat([train_x,test_x]))
#		reg.fit( tmp_df[0:len_train], train_y )
#		results = reg.predict( tmp_df[len_train:] )
		results =  tmp_df[:len_train].mean(axis=1)

		# this is not CV, as i am only evaluating over training-data
		# i need to bring the mean-function into CV to check
		# FIXME FIXME FIXME
#		from sklearn.metrics import r2_score
		scores=cross_val_score(reg, tmp_df[:len_train], train_y, cv=7, \
				scoring=qmetric)
		print("%s: %0.3f (+/- %0.2f) [Best]"%(qmetric,scores.mean(),\
				scores.std()))
#		print("r2: %.3f [Best-Mean]"%(r2_score( train_y, results)))
		results = tmp_df[len_train:].mean(axis=1)
	#	print tmp_df.head()
		0/0
	############

	ids = np.asarray( test_id )
	final_result = pd.DataFrame( {'Id':ids, target:results})
	print final_result.to_csv('final_stack_mean.csv',float_format='%d',index=False),
