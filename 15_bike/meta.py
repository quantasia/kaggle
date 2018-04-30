import pandas as pd
d = {
	'ds_classification' : False,		# otherwise regression
	'target' : 'count',
	'target_like' : ['registered','casual'],	# not in test-set
	'pred_id' : 'datetime',
	'ftrain' : 'data/train.csv',
	'ftest' : 'data/test.csv',
	'metric' : 'r2',
	'read' : pd.read_csv,
}
