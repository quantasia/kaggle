#!/usr/bin/env python
# v0.2
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Parameter
gridsearch=False

# Variables
target='Survived'
qcabin = True
qpclass = True
qfare = False
qembark = True
qtitle= True
qfamily = False

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data=pd.concat([train,test],ignore_index=True)

###################
#0) Preprocessing
###################

### Recode labels into integers
from sklearn.preprocessing import LabelEncoder
encs=[]
for v in ['Sex']:
	le = LabelEncoder()
	data[v] = data[v].fillna('NAN')
	le.fit(data[v].unique())
	data[v] = le.transform(data[v])
	encs.append(le)
#data['Sex'] = data['Sex'].replace('male',0)
#data['Sex'] = data['Sex'].replace('female',1)


# Shorten Namefield to the stated strings
#### over complete data
if True:
	import re
	for what in ['Mrs','Miss','Mr','Master','Dr','Rev']:
		reg = re.compile(r'(.*), %s.*'%what, flags=re.IGNORECASE)
		data['Name'] = data['Name'].str.replace( reg, what )
	reg = re.compile(r'(.*), .*', flags=re.IGNORECASE)
	data['Name'] = data['Name'].str.replace( reg, 'Unknown' )

	if qtitle:
		data['Title'] = data['Name'].replace(['Dr','Rev','Unkown'],'Rare')
		data['Title'] = data['Title'].map({"Master":0, "Miss":1, "Ms" : 1 ,\
								"Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
		data = pd.get_dummies(data,columns=['Title'])

	### replace names with average age
	data['AvgA']=data['Name']
	for what,nr in zip(['Miss','Master','Dr','Rev','Unknown','Mrs', 'Mr'],
								[21.77, 5.48, 43.57, 41.25, -1, 37, 32.25]):
		reg = re.compile(r'%s'%what, flags=re.IGNORECASE)
		data['AvgA'] = data['AvgA'].str.replace( what, str(nr) )

	data.loc[data['Age'].isnull(),'Age'] = data['AvgA'].astype(float)


# Fare
if qfare:
	data['Fare'] = np.log(data['Fare']+0.001)
else:
	data = data.drop('Fare',axis=1)

### Pclass
if qpclass:
	# make cabin a dummy variable attr[blue,2,3] into attr_1, attr_2 binaries
	data = pd.get_dummies(data, columns = ['Pclass'])

### Cabin
if qcabin:
	# replace all cabin-nrs by the first letter or, in case of NAN with x
	data['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' \
								for i in data['Cabin'] ])
#	data['Cabin'] = pd.Series(['Y' if not pd.isnull(i) else 'X' \
#								for i in data['Cabin'] ])

	# make cabin a dummy variable attr[blue,2,3] into attr_1, attr_2 binaries
	data = pd.get_dummies(data, columns = ['Cabin'])
else:
	data = data.drop('Cabin',axis=1)

### Embarkation
if qembark:
	data = pd.get_dummies(data, columns = ['Embarked'])
else:
	data = data.drop('Embarked',axis=1)

### Family
# Create a family size descriptor from SibSp and Parch
if qfamily:
	data["Fsize"] = data["SibSp"] + data["Parch"] + 1
	data['Single'] = data['Fsize'].map(lambda s: 1 if s == 1 else 0)
	data['SmallF'] = data['Fsize'].map(lambda s: 1 if  s == 2  else 0)
	data['MedF'] = data['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
	data['LargeF'] = data['Fsize'].map(lambda s: 1 if s >= 5 else 0)



data = data.fillna(0)
train_x=data[0:len(train)].drop(target,axis=1)
train_x=train_x.drop(['Ticket','Name','AvgA','PassengerId'],axis=1)
train_y=data[0:len(train)][target]
test_x=data[len(train):(len(train)+len(test))].drop(target,axis=1)
test_id=data[len(train):(len(train)+len(test))]['PassengerId']
test_x=test_x.drop(['Ticket','Name','AvgA','PassengerId'],axis=1)

print train_x.head()

#r = smf.ols('Survived ~ Age + Sex + Pclass + SibSp',data=train).fit()
#print r.summary()

######################
#1) Model formulation
######################

regs=[]
regs.append( LogisticRegression() )
regs.append( MLPClassifier(solver='lbfgs', alpha=1e-5, \
					hidden_layer_sizes=(20,7), random_state=1) )
regs.append( tree.DecisionTreeClassifier() )
regs.append( RandomForestClassifier(n_estimators=8) )
regs.append( AdaBoostClassifier(n_estimators=175,learning_rate=0.1) )
regs.append( GradientBoostingClassifier(n_estimators=100) )
regs.append( XGBClassifier(n_estimators=220) )
regs.append( VotingClassifier(estimators=[('log',regs[0]),('ann',regs[1]),\
	('randTree',regs[3]), ('adaboost',regs[4])], voting='hard'))
labels= ['Logistic Regression','Tree','RanForest', 'AdaBoost','GBRT',\
	'XGBoost', 'Voting']

####### GRID-Search:  finding best ANN param ##########
if gridsearch:
	mlp_grid=[ {'solver':['lbfgs'], 'hidden_layer_sizes':[(5, 2),\
					(10,2),(15,5),(20,5),(20,8),(25,12)]},\
				{'solver':['adam'], 'hidden_layer_sizes':[(10, 5),(20,7)]}]
	ada_grid = {'learning_rate':[0.01,0.1,0.2,0.3],\
				'n_estimators':[20,50,75,175,225]	} 
	reg_param = GridSearchCV(regs[4],ada_grid,cv=5,scoring='accuracy')
	reg_param.fit( train_x, train_y )
	print(reg_param.best_params_)
	means = reg_param.cv_results_['mean_test_score']
	stds = reg_param.cv_results_['std_test_score']
	for mean,std,params in zip(means, stds, reg_param.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"%(mean, std * 2, params))
	0/0
##########################################################

# Ensemble over all
print train_x.head()
from sklearn.model_selection import cross_val_score
for clf, label in zip( regs, labels ):
	scores = cross_val_score(clf, train_x, train_y, cv=7, scoring='accuracy')
	print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#####################################
# 5.0 Final prediction
####################################
0/0
reg=regs[1]
reg.fit( train_x, train_y )

results = reg.predict( test_x ).astype(int)
ids = np.asarray( test_id )
final_result = pd.DataFrame( {'PassengerId':ids, 'Survived':results})
final_result.to_csv('final.csv',index=False)
#df = pd.DataFrame( test[['PassengerId']], final_results )
#result_real = test[['Survived']]

