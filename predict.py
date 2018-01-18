import pandas as pd
import cPickle
from regressor import *
from sklearn.metrics import mean_squared_error

#train_data,train_score,data = cPickle.load(open('online_data.pkl'))
train_data, train_score, data = cPickle.load(open('online_data_final.pkl'))
regressor, scaler= cPickle.load(open('Lasso_Regressor.pkl'))

import numpy as np
mse = regressor.mse_path_[np.where(regressor.alphas_ == regressor.alpha_)]
print regressor.alpha_, mse.mean(), mse.max(), (regressor.coef_!=0).sum()

chosen_col = scaler.transform(train_data).columns[(regressor.coef_!=0)]
train_data[chosen_col].to_csv('explore/Lasso_train_data.csv')
scaler.transform(train_data)[chosen_col].to_csv('explore/Lasso_transformed_train_data.csv')
data[chosen_col].to_csv('explore/Lasso_data.csv')
scaler.transform(data)[chosen_col].to_csv('explore/Lasso_transformed_data.csv')

train_pred = regressor_predict(regressor, train_data, scaler)
pred_score = regressor_predict(regressor, data, scaler)

print 'Train MSE: %f' %mean_squared_error(train_score, train_pred)

output = pd.Series(pred_score, index=data.index)
output.to_csv('output_B.csv',header=None)