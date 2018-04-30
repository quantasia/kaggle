import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import numpy as np

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

r = smf.ols('Survived ~ Age + Pclass + SibSp',data=train).fit()
print r.summary()


reg = LogisticRegression()
print len(train)
train = train.fillna(0)	#FIXME
test = test.fillna(0)	#FIXME
#train = train.dropna( subset=['Age','Pclass','SibSp'] )		#FIXME
print len(train)


reg.fit( train[['Age','Pclass','SibSp']], train['Survived'] )

results = reg.predict( test[['Age','Pclass','SibSp']] )
ids = np.asarray( test['PassengerId'] )
final_result = pd.DataFrame( {'PassengerId':ids, 'Survived':results})
print final_result.to_csv(index=False)
#df = pd.DataFrame( test[['PassengerId']], final_results )
#result_real = test[['Survived']]

