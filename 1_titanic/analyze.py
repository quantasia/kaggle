#!/usr/bin/env python
# v0.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data=pd.concat([train,test],ignore_index=True)
target='Survived'

# Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age',
#       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'],
#     dtype='object')
print train.columns
print train.describe()


# 1) Target: Survived
print "Survival rate: ",train[target].mean()

# 2) Pclass
print train[[target,'Pclass']].groupby( ['Pclass'] ).mean()

# 3) Sex
print train[[target,'Sex']].groupby( ['Sex'] ).mean()

# 4) Age
#print sns.distplot( target[[target,'age']] )

####################
# example plots
#import seaborn as sns
#import matplotlib.pyplot as plt
#data['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' \
#                       for i in data['Cabin'] ])
#g = sns.factorplot(y='Survived',x="Cabin",data=data,kind="bar",order=['A',\
#                       'B','C','D','E','F','G','T','X'])
#g = g.set_ylabels("Survival Probability")
#plt.show()

# 5) qembark
print train[[target,'Embarked']].groupby( ['Embarked'] ).mean()
