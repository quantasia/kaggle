#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import RandomForestRegressor 
#from sklearn.neural_network import MLPRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from xgboost import XGBRegressor

#from sklearn.model_selection import cross_val_score


#################
# 0) Parameters
################
target='requester_received_pizza'

train=pd.read_json('data/train.json') 
test=pd.read_json('data/test.json')

########################
# 0)  Analyze attributes:  distributation, nan, etc.
########################

#### 0.1 Summary over numeric features ####
#train.describe().to_csv('summary.csv',float_format='%.2f')

ltext = []
lclass = []
train_text = train.select_dtypes(include=['object'])
for t in train_text.columns:
	print t
	if type(train_text[t][0]) in [str, unicode]:
		if train_text[t][0:100].value_counts().sum() > 90:
			ltext = ltext + [t]
		else:
			lclass = lclass + [t]
	else:
		print "Error:  Other datatype than str or unicode"
print ltext
print lclass
#### 0.2 Summary over text features ####
#train.describe().to_csv('summary.csv',float_format='%.2f')

#### 0.3 Summary over class features ####
#train.describe().to_csv('summary.csv',float_format='%.2f')

#### 0.4 Summary over ordinal features ####
#train.describe().to_csv('summary.csv',float_format='%.2f')

0/0

#########################
# 1) Analyze attributes relationship
#########################

## how many is target variable
print train[target].mean()

text_features = ['request_title', 'request_text','giver_username_if_known',\
'request_id','request_text_edit_aware','requester_subreddits_at_request',\
'requester_user_flair','requester_username']
train = train.drop(text_features,axis=1)

print "Correlation to target variable"
print	train.corrwith(train[target]).sort_values()






0/0
data=pd.concat([train,test],ignore_index=True)
target='SalePrice'

# FULL
qexp_0=['OverallQual','OverallCond','YearBuilt','1stFlrSF','2ndFlrSF',\
			'YearRemodAdd','LotArea','GrLivArea', 'LotFrontage' ,\
			'LotArea','MasVnrArea','BsmtFinSF1', 'BsmtFinSF2',\
			'TotalBsmtSF','TotRmsAbvGrd','MiscVal','YrSold']
# Group 1: category to number		[good, ok, bad] -> [2, 1, 0]
qexp_1=['Utilities','BsmtExposure','BsmtFinType1','LotShape',\
'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',\
'CentralAir','Electrical','Functional','PavedDrive' ] 
#,'Fence','MSSubClass']
# Group 2: dummies		[blue, yellow, pink] -> blue=1/0, yellow=1/0, ...
qexp_2=['Neighborhood','HouseStyle','BldgType','MiscFeature',\
'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', \
'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'RoofStyle',\
'Exterior1st','Exterior2nd','MasVnrType','SaleType','SaleCondition']


# Group 0:	No Formatting necessary
prev="""
qexp_0=['OverallQual','OverallCond','YearBuilt','1stFlrSF','2ndFlrSF',\
			'YearRemodAdd','LotArea','GrLivArea']#,'HouseStyle',]
# Group 1: category to number		[good, ok, bad] -> [2, 1, 0]
qexp_1=['Utilities','BsmtExposure','BsmtFinType1']  #,'Fence','MSSubClass']
# Group 2: dummies		[blue, yellow, pink] -> blue=1/0, yellow=1/0, ...
qexp_2=['Neighborhood','HouseStyle','BldgType','MiscFeature']
"""

qexp=qexp_0+qexp_1+qexp_2

###################
#1 Prepare data
###################
all_vars=[u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
u'Street', u'Alley', u'LotShape', u'LandContour', u'Utilities',
u'LotConfig', u'LandSlope', u'Neighborhood', u'Condition1',
u'Condition2', u'BldgType', u'HouseStyle', u'OverallQual',
u'OverallCond', u'YearBuilt', u'YearRemodAdd', u'RoofStyle',
u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
u'MasVnrArea', u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual',
u'BsmtCond', u'BsmtExposure', u'BsmtFinType1', u'BsmtFinSF1',
u'BsmtFinType2', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF',
u'Heating', u'HeatingQC', u'CentralAir', u'Electrical', u'1stFlrSF',
u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
u'KitchenAbvGr', u'KitchenQual', u'TotRmsAbvGrd', u'Functional',
u'Fireplaces', u'FireplaceQu', u'GarageType', u'GarageYrBlt',
u'GarageFinish', u'GarageCars', u'GarageArea', u'GarageQual',
u'GarageCond', u'PavedDrive', u'WoodDeckSF', u'OpenPorchSF',
u'EnclosedPorch', u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'PoolQC',
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
#for v in qexp_2:
#	data = pd.get_dummies(data,columns=[v])
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

##################################
### Analyze ###
##################################

### Group 0:  Single Attributes ###
import seaborn as sns
import matplotlib.pyplot as plt

if False:
	import seaborn as sns
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(ncols=2,nrows=2)
	y=-1
	for var,i in zip(qexp_0[0:5],range(0,4)):
		x = 1 if i%2==0 else 0
		y = y+1 if i%2==0 else y
		print var,i,x,y
		sns.distplot(data[var],ax=axs[y][x])
	
	plt.show()


#### check distribution value -> with boxplots (!)

from sklearn.preprocessing import RobustScaler

print qexp_0

var='GrLivArea'
sns.distplot(data[var])
plt.show()
print data[var][0:10]
robust_scaler = RobustScaler()
sns.distplot( robust_scaler.fit_transform( np.array(data[var]).reshape(-1,1)) )
plt.show()
0/0
print data[var][0:10]



#for var in qexp_0:
#	if len( data[var].unique() ) < 10:
#		print data[[var,target]].groupby(var).agg({target:['count','mean']} )




#for var in qexp_1:
#	if len( data[var].unique() ) < 10:
#		print data[[var,target]].groupby(var).agg( {target:['count','mean']})


#var='Neighborhood'
#print data[[var,target]].groupby(var).agg( {target:['count','mean']})
#0/0





######################################
# Single variable prediction score
######################################
all_vars = []

#train_x = pd.DataFrame()
#train_x['OverallQual'] = data[['OverallQual']][0:len(train)]
#train_x['Neighborhood'] = data[['Neighborhood']][0:len(train)]
#train_x = pd.get_dummies(train_x,\
#			columns=['Neighborhood'])[0:len(train)]
#print train_x[0:1]
#0/0
for var in qexp_0+qexp_1+qexp_2:

	train_x = pd.DataFrame()
#	train_x['OverallQual'] = data[['OverallQual']][0:len(train)]
#	train_x['GrLivArea'] = data[['GrLivArea']][0:len(train)]
#	train_x['BsmtFinSF1'] = data[['BsmtFinSF1']][0:len(train)]
#	train_x['YearBuilt'] = data[['YearBuilt']][0:len(train)]
#	train_x['2ndFlrSF'] = data[['2ndFlrSF']][0:len(train)]
#	train_x['YearRemodAdd'] = data[['YearRemodAdd']][0:len(train)]
#	train_x['TotalBsmtSF'] = data[['TotalBsmtSF']][0:len(train)]
#	train_x['BsmtFinType1'] = data[['BsmtFinType1']][0:len(train)]
#	train_x['Functional'] = data[['Functional']][0:len(train)]

	# special treatment for vars in group 2
	train_x[var] =  data[[var]][0:len(train)]
	if var in qexp_2:
		train_x = pd.get_dummies(train_x,columns=[var])[0:len(train)]
		qtype=2
	else:
		qtype = 1 if var in qexp_1 else 0

	regs = [XGBRegressor(), LinearRegression(), RandomForestRegressor() ]
	infos=['XGB','LinReg','RF']

	# loop over all regressors
	best_reg = np.array(-999)
	best_info = None
	for reg,info in zip(regs,infos):
		scores=cross_val_score(reg, train_x, train_y, cv=4, scoring='r2')
#		print("R2: %0.3f (+/- %0.2f) [%s]"%(scores.mean(),scores.std(),info))
		if scores.mean() >  best_reg.mean():
			best_reg = scores
			best_info = info
	all_vars.append( [var, best_reg.mean(),best_reg.std(),best_info,qtype] )
	
df_vars = pd.DataFrame(all_vars)
df_vars.columns = ['Variable','Mean','Std','Method','Vartype']
print df_vars.sort_values(by=['Mean'],ascending=False)[0:10]


###### 
# Best variable combination prediction score
######
# TODO
0/0
for i in range(0,10):
	predict_vars = []
	for i2 in range(0,i):
		all_vars[i][0] # name
		# special treatment for vars in group 2
		if var in qexp_2:
			train_x = pd.get_dummies(data[[var]],columns=[var])[0:len(train)]
		else:
			train_x[var] =  data[[var]][0:len(train)]
	scores=cross_val_score(reg, train_x, train_y, cv=3, scoring='r2')
	
#	for var in all_vars:
#	print var[0] #varname

# TODO
