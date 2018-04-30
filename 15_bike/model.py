#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import meta
if meta.d['ds_classification']:			# classification
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import RandomForestClassifier 
	from sklearn.ensemble import GradientBoostingClassifier
	from xgboost import XGBClassifier
else:				# regression
	from sklearn.linear_model import LinearRegression
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.ensemble import AdaBoostRegressor
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import GradientBoostingRegressor
	from xgboost import XGBRegressor

#################
# 0) Parameters
################

grid_search=False

train= meta.d['read']( meta.d['ftrain'] ) 
len_train=len(train)
test= meta.d['read']( meta.d['ftest'] )
data=pd.concat([train,test],ignore_index=True)
target='count'
pred_id='datetime'

num_feature =train.select_dtypes(include=['int64','float']).columns.tolist()
text_feature = []			# essay
class_feature = []		# blue, yellow, pink -> dummies
ordinal_feature = []		# good, ok, bad		-> number

if target in num_feature:
	num_feature.remove( target )
all_feature=num_feature+text_feature+class_feature+ordinal_feature+[target]

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

data['year'] = pd.to_datetime(data['datetime']).dt.year
data['month'] = pd.to_datetime(data['datetime']).dt.month
data['hour'] = pd.to_datetime(data['datetime']).dt.hour

### dummy adding
for v in class_feature:
	data = pd.get_dummies(data,columns=[v])
#	data=data.drop('Neighborhood',axis=1)

data = data.fillna(0)
test_id=data[len(train):(len(train)+len(test))][pred_id]
data=data.drop(pred_id,axis=1)
test_x=data[len(train):(len(train)+len(test))].drop(target,axis=1)
## remove target from train_x,  also other columns not in test-set
train_x=data[0:len(train)].drop(target,axis=1)
if len(meta.d['target_like']) > 0:
	for tar in meta.d['target_like']:
		if tar in train_x.columns:
			train_x=train_x.drop(tar,axis=1)
		if tar in test_x.columns:
			test_x=test_x.drop(tar,axis=1)		# filled with na due to data
train_y= data[0:len(train)][target]
##################
#2 Model data
###################

#import sys
#sys.path.insert(0, '../helper')
#from meta_predictor import BestRegressor

#meta = BestRegressor(train_x, train_y, 4, 'r2', 0)
#meta.evaluate()


regs=[]
regs.append( DecisionTreeRegressor() )
regs.append( AdaBoostRegressor(n_estimators=120,learning_rate=0.2) )
regs.append( RandomForestRegressor(n_estimators=50,max_depth=8) )
regs.append( GradientBoostingRegressor(n_estimators=150,max_depth=3) )
regs.append( XGBRegressor(n_estimators=275,max_depth=3,\
						early_stopping_rounds=5) )
infos=[]
for r in regs:
	infos.append( r.__class__.__name__ )

train_y= train_y.astype(int)
for reg,info in zip(regs,infos):
	scores=cross_val_score(reg, train_x, train_y, cv=7, \
									scoring=meta.d['metric'])
	print("%s: %0.3f (+/- %0.2f) [%s]"%(meta.d['metric'],scores.mean(),\
									scores.std(),info))


##################
#5 Predict data
###################
if True:
	
	reg=regs[4]
	reg.fit(train_x,train_y)
	results = reg.predict( test_x ).astype(int).clip(1,9999999)

	ids = np.asarray( test_id )
	final_result = pd.DataFrame( {pred_id:ids, target:results})
	print(final_result.describe())
	final_result[[pred_id,target]].to_csv('pred1.csv',float_format='%d',index=False),
