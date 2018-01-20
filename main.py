import cPickle

train_data, train_score, test_data = cPickle.load(open('online_data.pkl'))
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
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LassoCV
import numpy as np

regressor = LassoCV(normalize=False, n_jobs=-1, alphas=np.arange(0.0001,0.010,0.0002), cv=ShuffleSplit(n_splits=30,test_size=0.2))

regressor, scaler = regressor_train(regressor, train_data, train_score, auxiliary_data=None, test_data=test_data,
                                    normalize='categorical', discrete_col=new_col_name, max_z_score=2, discrete_max_z_score=2.5,
                                    discrete_weight=5)

from sklearn.metrics import mean_squared_error
print mean_squared_error(train_score, regressor_predict(regressor, train_data, scaler))

import numpy as np
mse = regressor.mse_path_[np.where(regressor.alphas_ == regressor.alpha_)]
print regressor.alpha_,mse.mean(), mse.max(), (regressor.coef_!=0).sum()

# Dump the model and column name
cPickle.dump((train_data, train_score, test_data) , open('online_data_final.pkl','w'))
cPickle.dump((scaler.transform(train_data), train_score, scaler.transform(test_data)) , open('online_data_upload.pkl','w'))
cPickle.dump((regressor, scaler), open('Lasso_Regressor.pkl','w'))