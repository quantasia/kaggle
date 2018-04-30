#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#################
# 0) Parameters
################

grid_search=False

train=pd.read_json('data/train.json') 
len_train=len(train)
test=pd.read_json('data/test.json')
data=pd.concat([train,test],ignore_index=True)
target='requester_received_pizza'
pred_id='request_id'

num_feature = ['requester_upvotes_plus_downvotes_at_retrieval',\
'request_number_of_comments_at_retrieval',\
'requester_number_of_comments_in_raop_at_retrieval',\
'requester_number_of_posts_on_raop_at_request',\
'requester_number_of_comments_in_raop_at_request']
text_feature = []			# essay
class_feature = []		# blue, yellow, pink -> dummies
ordinal_feature = []		# good, ok, bad		-> number


#qexp_0=['OverallQual','GrLivArea','TotalBsmtSF','1stFlrSF','YearBuilt',\
#	'LotArea','LotFrontage']
# category to number		[good, ok, bad] -> [2, 1, 0]
#qexp_1=['ExterQual','BsmtQual','BsmtFinType1','BsmtExposure']
# dummies		[blue, yellow, pink] -> blue=1/0, yellow=1/0, ...
#qexp_2=['Neighborhood','MSSubClass']

all_feature=num_feature+text_feature+class_feature+ordinal_feature+[target]
#qexp=qexp_0+qexp_1+qexp_2
###################
#1 Prepare data
###################
for v in train.columns:
	if v not in all_feature+[pred_id]:
		data=data.drop(v,axis=1)

### ordinal features
from sklearn.preprocessing import LabelEncoder
encs=[]
for v in ordinal_feature:
	le = LabelEncoder()
	data[v] = data[v].fillna('NAN')
	le.fit(data[v].unique())
	data[v] = le.transform(data[v])
	encs.append(le)

### dummy adding
for v in class_feature:
	data = pd.get_dummies(data,columns=[v])
#	data=data.drop('Neighborhood',axis=1)

data = data.fillna(0)
test_id=data[len(train):(len(train)+len(test))][pred_id]
data=data.drop(pred_id,axis=1)
train_x=data[0:len(train)].drop(target,axis=1)
train_y= data[0:len(train)][target]
test_x=data[len(train):(len(train)+len(test))].drop(target,axis=1)

##################
#2 Model data
###################

#import sys
#sys.path.insert(0, '../helper')
#from meta_predictor import BestRegressor

#meta = BestRegressor(train_x, train_y, 4, 'r2', 0)
#meta.evaluate()


regs=[]
regs.append( DecisionTreeClassifier() )
regs.append( AdaBoostClassifier(n_estimators=120,learning_rate=0.2) )
regs.append( RandomForestClassifier(n_estimators=50,max_depth=8) )
regs.append( GradientBoostingClassifier(n_estimators=150,max_depth=3) )
regs.append( XGBClassifier(n_estimators=275,max_depth=3,\
						early_stopping_rounds=5) )
infos=[]
for r in regs:
	infos.append( r.__class__.__name__ )


train_y= train_y.astype(bool)
qmetric='accuracy'
for reg,info in zip(regs,infos):
	scores=cross_val_score(reg, train_x, train_y, cv=7, scoring=qmetric)
	print("%s: %0.3f (+/- %0.2f) [%s]"%(qmetric,scores.mean(),scores.std(),\
			info))


##################
#5 Predict data
###################
if True:
	
	reg=regs[0]
	reg.fit(train_x,train_y)
	results = reg.predict( test_x ).astype(int)

	ids = np.asarray( test_id )
	final_result = pd.DataFrame( {pred_id:ids, target:results})
	print(final_result.describe())
#	final_result.to_csv('pred1.csv',float_format='%d',index=False),
