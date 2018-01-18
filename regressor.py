from data_scaling import *

# Training regressor with cross validation option and scaler option
def regressor_train(rgs, train_data, train_score, auxiliary_data=None, test_data=None, normalize=False, pca_n_components=None,
                    discrete_col=[],extreme_process='shrink', max_z_score=4, discrete_max_z_score=4, discrete_weight=1,
                    cv=None, **cv_params):

	scaler = YGTQ_Scaler(method=normalize, pca_n_components=pca_n_components, max_z_score=max_z_score, discrete_col=discrete_col,
	                     extreme_process=extreme_process, discrete_max_z_score=discrete_max_z_score, discrete_weight=discrete_weight)

	train_data, train_score = scaler.fit_transform(train_data, train_score, auxiliary_data=auxiliary_data, test_data=test_data)

	print train_data.shape

	if cv:
		from sklearn.model_selection import GridSearchCV
		rgs = GridSearchCV(rgs, cv=cv, n_jobs=-1, **cv_params)

	rgs.fit(train_data, train_score)

	from sklearn.metrics import mean_squared_error
	print 'Train MSE: %f' % mean_squared_error(train_score, rgs.predict(train_data))

	return rgs, scaler


def regressor_predict(rgs, test_data, scaler):

	test_data = scaler.transform(test_data)
	return scaler.y_scaler.inverse_transform(rgs.predict(test_data))


if __name__ == '__main__':
	import pandas as pd
	from sklearn.linear_model import Lasso
	train_data = pd.DataFrame([[1,2,3],[2,3,1]])
	rgs = Lasso()
	rgs, X_scaler = regressor_train(rgs, train_data, [1,1], normalize=True)
	print X_scaler