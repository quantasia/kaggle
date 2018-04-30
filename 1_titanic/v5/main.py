#!/usr/bin/env python
# v0.2
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


qexp = ['Age','Pclass','SibSp', 'Parch']#, 'Fare' ]
qexp_f = ['Sex']#,'Embarked']
qexp = qexp+qexp_f

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data=pd.concat([train,test],ignore_index=True)

###################
#0) Preprocessing
###################

### Recode labels into integers
from sklearn.preprocessing import LabelEncoder
encs=[]
for v in qexp_f:
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

	print data[['Name','Age']].groupby('Name').agg({'Name':'count','Age':'mean'})

	### replace names with average age
	data['AvgA']=data['Name']
	for what,nr in zip(['Miss','Master','Dr','Rev','Unknown','Mrs','Mr'],
	#							[22, 5,44, 41, -1, 37, 32]):#42,94
								[21.77, 5.48,43.57, 41.25, -1, 36.99, 32.25]):#42,94
	#							[21, 5, 43, -1, -1, 36, 32]):#42,94
		reg = re.compile(r'%s'%what, flags=re.IGNORECASE)
		data['AvgA'] = data['AvgA'].str.replace( what, str(nr) )
	#	print data['Name']
	#	0/0

	data.loc[data['Age'].isnull(),'Age'] = data['AvgA'].astype(float)

data['Fare'] = np.log(data['Fare']+0.001)
data = data.fillna(0)

train=data[0:len(train)]
test=data[len(train):(len(train)+len(test))]

#r = smf.ols('Survived ~ Age + Sex + Pclass + SibSp',data=train).fit()
#print r.summary()

######################
#1) Model formulation
######################

reg1 = LogisticRegression()
reg2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10), random_state=1)
reg3 = tree.DecisionTreeClassifier()
reg4 = RandomForestClassifier()
reg5 = AdaBoostClassifier(n_estimators=100)
reg6 = GradientBoostingClassifier()
reg = VotingClassifier(estimators=[('log', reg1), ('ann', reg2), \
		('tree',reg3), ('randTree',reg4), ('adaboost',reg5)], voting='hard')
target='Survived'
reg=reg2

####### GRID-Search:  finding best ANN param ##########
s="""
tuned_parameters=[ {'solver':['lbfgs'], 'hidden_layer_sizes':[(5, 2)]},\
						{'solver':['lbfgs'], 'hidden_layer_sizes':[(10, 5)]},\
						{'solver':['lbfgs'], 'hidden_layer_sizes':[(10, 7)]},\
						{'solver':['adam'], 'hidden_layer_sizes':[(10, 5)]} 

						]
from sklearn.model_selection import GridSearchCV
reg_param = GridSearchCV(reg2, tuned_parameters, cv=5,
                       scoring='accuracy')
reg_param.fit( train[qexp], train[target] )
print(reg_param.best_params_)
means = reg_param.cv_results_['mean_test_score']
stds = reg_param.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, reg_param.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r"%(mean, std * 2, params))
0/0
"""
##########################################################

# Ensemble over all
from sklearn.model_selection import cross_val_score
for clf, label in zip([reg1,reg2, reg3, reg4, reg5, reg6, reg], 
	['Logistic Regression','ANN','Forest', 'Ensemble','RandomForest',\
	'Adaboost','Gradient']):
	scores = cross_val_score(clf, train[qexp], train[target], cv=5, scoring='accuracy')
	print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#print cross_val_score(reg, train[qexp], train[target], cv=5, scoring='accuracy').mean()

#from sklearn.model_selection import KFold
#kf = KFold(n_splits=4)
#avgs=0
#for itrain, itest in kf.split(train):
#	kftrainx = np.array(train[qexp])[itrain]
#	kftrainy = np.array(train['Survived'])[itrain]
#	kftestx = np.array(train[qexp])[itest]	
#	kftesty = np.array(train['Survived'])[itest]	
#	reg.fit(kftrainx, kftrainy )
#	kfresult = reg.predict( kftestx )
#	print accuracy_score( kftesty, kfresult )
#	avgs = avgs + accuracy_score( kftesty, kfresult )
#print "Average all folds: ",avgs/4.0
0/0
reg.fit( train[ qexp ], train['Survived'] )

results = reg.predict( test[ qexp ] ).astype(int)
ids = np.asarray( test['PassengerId'] )
final_result = pd.DataFrame( {'PassengerId':ids, 'Survived':results})
print final_result.to_csv(index=False)
#df = pd.DataFrame( test[['PassengerId']], final_results )
#result_real = test[['Survived']]

