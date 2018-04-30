#!/usr/bin/env python
# v0.2
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#0) Preprocessing
train = train.fillna(0)	#FIXME
test = test.fillna(0)	#FIXME

train['Sex'] = train['Sex'].replace('male',0)
train['Sex'] = train['Sex'].replace('female',1)
test['Sex'] = test['Sex'].replace('male',0)
test['Sex'] = test['Sex'].replace('female',1)

#r = smf.ols('Survived ~ Age + Sex + Pclass + SibSp',data=train).fit()
#print r.summary()

qexp = ['Age','Pclass','Sex','SibSp']

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

reg1 = LogisticRegression()
reg2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1)
reg3 = tree.DecisionTreeClassifier()
#reg = RandomForestClassifier()
reg = VotingClassifier(estimators=[('log', reg1), ('ann', reg2), ('tree',\
				reg3)], voting='hard')
target='Survived'

####### GRID-Search:  finding best ANN param ##########
s="""
tuned_parameters=[ {'solver':['lbfgs'], 'hidden_layer_sizes':[(5, 2)]},\
						{'solver':['lbfgs'], 'hidden_layer_sizes':[(10, 5)]},\
						{'solver':['lbfgs'], 'hidden_layer_sizes':[(10, 7)]},\
						{'solver':['lbfgs'], 'hidden_layer_sizes':[(15, 10)]},\
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
0/0"""
##########################################################

# Ensemble over all
from sklearn.model_selection import cross_val_score
for clf, label in zip([reg1,reg2, reg3, reg], ['Logistic Regression','ANN',\
			'Forest', 'Ensemble']):
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
#0/0
reg.fit( train[ qexp ], train['Survived'] )

results = reg.predict( test[['Age','Pclass','Sex','SibSp']] )
ids = np.asarray( test['PassengerId'] )
final_result = pd.DataFrame( {'PassengerId':ids, 'Survived':results})
print final_result.to_csv(index=False)
#df = pd.DataFrame( test[['PassengerId']], final_results )
#result_real = test[['Survived']]

