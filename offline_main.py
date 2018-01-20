import cPickle
import pandas as pd

train_data, train_score, test_data, test_score = cPickle.load(open('offline_data.pkl'))
print train_data.shape, test_data.shape

import re
def _identify_categorical_variable(df):
	tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
	categorical_columns = filter(lambda x: re.match(tool_mark, str(x)), df.columns)
	#return categorical_columns
	return ['TOOL', 'Tool', 'TOOL_ID', 'Tool (#1)', 'TOOL (#1)', 'TOOL (#2)', 'Tool (#2)', 'Tool (#3)', 'Tool (#4)',
	 'OPERATION_ID','Tool (#5)', 'TOOL (#3)']

from categorical_processing import *
full_data = pd.concat([train_data, test_data], axis=0)
discrete_col = _identify_categorical_variable(full_data)
categorical_data = full_data.loc[:,discrete_col]
categorical_data.columns = map(lambda x: x+'_new', categorical_data.columns)
full_data = pd.concat([full_data, categorical_data], axis=1)
full_data, enc, mapping_dict, new_col_name = categorical_encoding(full_data, categorical_data.columns)
new_col_name = filter(lambda x: not re.match(r'.*\_0',x), new_col_name)
full_data.drop(labels=filter(lambda x: re.match(r'.*\_0',x), full_data.columns), axis=1,inplace=True)

train_data = full_data.loc[train_data.index,]
test_data = full_data.loc[test_data.index,]

# Train the model
from regressor import *
from sklearn.linear_model import Lasso, LassoCV
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit

scaler = YGTQ_Scaler(method='categorical', max_z_score=3, discrete_col=new_col_name, discrete_max_z_score=2,
                     discrete_weight=10)

train_data, train_score = scaler.fit_transform(train_data, train_score, auxiliary_data=None, test_data=test_data)
test_data = scaler.transform(test_data)

regressor = LassoCV(normalize=False, alphas=np.arange(0.0001,0.010,0.0002),cv=ShuffleSplit(n_splits=30,test_size=0.2),n_jobs=-1)
regressor.fit(train_data, train_score)
import numpy as np
estimator = regressor
mse = estimator.mse_path_[np.where(estimator.alphas_ == estimator.alpha_)]
print regressor.alpha_, mse.mean(), mse.max(), (estimator.coef_!=0).sum()

chosen_col = train_data.columns[(regressor.coef_!=0)]
train_data[chosen_col].to_csv('explore/Lasso_train_data.csv')
test_data[chosen_col].to_csv('explore/Lasso_data.csv')


for alpha in np.arange(0.001,0.01,0.001):
	regressor = Lasso(normalize=False, alpha=alpha)
	regressor.fit(train_data, train_score)

	train_pred = scaler.y_scaler.inverse_transform(regressor.predict(train_data))
	pred_score = scaler.y_scaler.inverse_transform(regressor.predict(test_data))

	print alpha
	print 'Train MSE: %f' % mean_squared_error(scaler.y_scaler.inverse_transform(train_score), train_pred)
	print 'Test MSE: %f' % mean_squared_error(test_score, pred_score)

