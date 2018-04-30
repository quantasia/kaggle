#!/usr/bin/env python
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import sent_tokenize, word_tokenize
from string import punctuation
import tqdm

#################
# 0) Parameters
################

grid_search=False

train=pd.read_json('data/train.json') 
train=train[0:1000]
len_train=len(train)
test=pd.read_json('data/test.json')
data=pd.concat([train,test],ignore_index=True)
target='requester_received_pizza'
pred_id='request_id'

text_feature = 'request_text'			# essay
class_feature = []		# blue, yellow, pink -> dummies
ordinal_feature = []		# good, ok, bad		-> number


for v in data.columns:
	if v not in [target]+[text_feature]:
		data = data.drop(v,axis=1)
print data.head()

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 


def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]


def preprocess_text(text):
	# Input: str, i.e. document/sentence
	# Output: list(str) , i.e. list of lemmas
	return [	word for word in lemmatize_sent(text)
				if word not in stoplist_combined
				and not word.isdigit()]

stopwords_punct = set(punctuation)
stoplist_combined = stopwords_punct
print("Preprocessing ...")
count_vect = CountVectorizer(analyzer=preprocess_text)
print("Fitting ...")
train_set = count_vect.fit_transform(train[text_feature])
print type(train_set)
print train_set.head()

0/0


all_feature=num_feature+text_feature+class_feature+ordinal_feature+[target]


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
