# coding=utf-8
import pandas as pd
import re


def data_preprocessing(df, DROP_THRESHOLD=None, mean_recond=None, by_category=False, redundancy_process=False, check=False,
                       zero_equal_na=False, augment=False, remove_correlation=False):

	# 目前在by_category选项下不支持训练集和测试集分开填充缺失值
	assert mean_recond is None if by_category else True
	# 识别0为异常值的函数待改进
	assert not zero_equal_na

	def __identify_categorical_variable(df):
		# 识别工具变量
		tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
		categorical_columns = filter(lambda x: re.match(tool_mark, x), df.columns)
		return categorical_columns

	def __identify_date(df):
		# 识别日期变量
		date_column_drop = []
		for index, i in enumerate(df.columns):
			try:
				if re.match(r'20[0,1]\d[0,1]\d[0,1,2,3]\d{1,}', str(long(df[i][df[i].notnull()][0]))):
					# if str(long(df.iloc[0, index])).startswith('2017'):
					date_column_drop.append(i)
			except ValueError:
				pass
		return date_column_drop

	def __create_nan(df, median_thd=2):
		# 0是某一列的最大或最小值，并且该维度的中位数超过一定的阈值，替换为缺失值
		import numpy as np
		lower_bool = df.apply(lambda x: x.min() == 0 and x.median() > median_thd)
		df.loc[:, df.columns[lower_bool]] = df.loc[:, df.columns[lower_bool]].replace(0, np.nan)
		upper_bool = df.apply(lambda x: x.max() == 0 and x.median() < -median_thd)
		df.loc[:, df.columns[upper_bool]] = df.loc[:, df.columns[upper_bool]].replace(0, np.nan)
		return df

	def _data_augmentation(df):
		# 相邻两列之差作为新的特征，条件是两个维度在同一数量级下
		augmented_df = df
		for i in xrange(df.shape[1] - 1):
			first_col = df.iloc[:, i]
			last_col = df.iloc[:,i + 1]
			if last_col.mean() / first_col.mean() > 10 or last_col.mean() / first_col.mean() < 0.1:
				continue
			else:
				augmented_df = pd.concat(
					[augmented_df, pd.Series(last_col - first_col, name=last_col.name + '-' + first_col.name)],
					axis=1)
		return augmented_df

	if zero_equal_na:
		df = __create_nan(df)

	if mean_recond is None:
		print 'Train_data preprocessing...'
		print df.shape
		# 此时传入train_data
		# 删除表中全部为NaN的列
		df.dropna(axis=1, how='all', inplace=True)
		print df.shape
		# 记录分类变量
		categorical_columns = __identify_categorical_variable(df)
		if check:
			df.loc[:,categorical_columns].to_csv('explore/check_categorical.csv')
		# 丢弃日期变量
		date_columns = __identify_date(df)
		if check:
			df.loc[:,date_columns].to_csv('explore/check_date.csv')
		df.drop(labels=date_columns, axis=1, inplace=True)
		print df.shape

		# 丢弃众数占比高于阈值的特征，跳过分类变量
		column_drop = []
		for i in set(df.columns)-set(categorical_columns):
			if df[i][df[i] == df[i].mode()[0]].shape[0] >= DROP_THRESHOLD * (df.shape[0]-df[i].isnull().sum()):
				column_drop.append(i)
		if check:
			df.loc[:,column_drop].to_csv('explore/check_threshold.csv')
		df = df.drop(labels=column_drop, axis=1)
		print df.shape

		if not by_category:
			#用训练集的均值填充缺失值
			mean_recond = df.mean()
			df = df.fillna(mean_recond,inplace=True)
			#用众数
			#df.fillna(df.mode().iloc[0],inplace=True)
			return df,mean_recond
		else:
			import categorical_processing
			# 特征按照工具变量分块
			feature_dict = categorical_processing.feature_subgrouping(df, categorical_columns)
			final_df = pd.DataFrame(index=df.index)
			redundancy_dict = {}
			for category in categorical_columns:
				partial_df = categorical_processing.chunk_dataframe_generator(df, feature_dict, category)
				# 使用工具变量取值相同的数据的中位数填充缺失值
				partial_df = partial_df.groupby(category).apply(lambda x: x.fillna(x.median())).reset_index(level=0, drop=True)
				# 如果一个特征在工具变量的某个取值下全部为缺失值，丢弃该特征
				partial_df = partial_df.dropna(axis=1, how='any')
				if augment:
					# 相邻两个特征的差作为新的特征
					partial_df = pd.concat([partial_df[category], _data_augmentation(partial_df.iloc[:, 1:])], axis=1)
				if remove_correlation:
					# 相邻特征的相关度超过阈值的，采取指定的处理方法
					from dimensionality_reduction import correlation_remove
					partial_df = correlation_remove(partial_df, threshold=0.85, method='remove')
				if redundancy_process:
					# 同一工具变量下的冗余维度移除
					from dimensionality_reduction import redundancy_processing
					partial_df, result_dict = redundancy_processing(partial_df)
					redundancy_dict.update(result_dict)
					if check:
						result_df = pd.DataFrame()
						for col in result_dict.itervalues():
							result_df = pd.concat([result_df, df.loc[:, col]], axis=1)
						result_df.to_csv('explore/check_redundancy.csv')
				final_df = pd.concat([final_df, partial_df], axis=1)
			return final_df

	else:
		print 'Test_data preprocessing...'

		#此时传入test_data
		#用全部样本填充缺失值
		df = df.fillna(mean_recond,inplace=True)
		return df

def data_split(data,mode=None,DROP_THRESHOLD=None,by_category=False,redundancy_process=False, check=False,
                       zero_equal_na=False, augment=False, remove_correlation=False):
	# Train data and test data split
	assert mode in ['online','offline']
	print 'Data spliting... in %s mode' %mode

	if mode == 'offline':
		# 线下测试
		test_data = pd.read_csv('test_A.csv', index_col=0)
		test_score = pd.read_csv('test_A_score.csv',index_col=0,header=None)
		train_score = data.Y
		train_data = data.drop(labels='Y', axis=1)
		assert train_data.shape[1] == test_data.shape[1]
		assert test_data.shape[0] == len(test_score)

		# 训练数据和测试数据共同预处理，之后再分离
		full_data = pd.concat([train_data, test_data], axis=0)
		full_data = data_preprocessing(full_data, DROP_THRESHOLD=DROP_THRESHOLD, redundancy_process=redundancy_process, by_category=by_category,
		                          zero_equal_na=zero_equal_na, augment=augment, remove_correlation=remove_correlation, check=check)

		test_data = full_data.loc[test_data.index,:]
		train_data = full_data.loc[train_data.index,:]

		print train_data.shape, test_data.shape

		if check:
			train_data.to_csv('explore/train_data_for_explore.csv')
			test_data.to_csv('explore/test_data_for_explore.csv')

		return train_data,train_score,test_data,test_score

	elif mode == 'online':
		# 线上测试
		train_score = pd.concat([data.Y, pd.read_csv('test_A_score.csv',index_col=0,header=None,names=['Y']).Y])
		train_data = pd.concat([data.drop(labels='Y', axis=1),pd.read_csv('test_A.csv', index_col=0)], axis=0)

		test_data = pd.read_csv('final_test_B.csv', index_col=0)
		assert train_data.shape[1] == test_data.shape[1]
		assert train_data.shape[0] == len(train_score)

		# 训练数据和测试数据共同预处理，之后再分离
		full_data = pd.concat([train_data, test_data], axis=0)
		full_data = data_preprocessing(full_data, DROP_THRESHOLD=DROP_THRESHOLD, redundancy_process=redundancy_process,
		                               by_category=by_category, zero_equal_na=zero_equal_na, augment=augment,
		                               remove_correlation=remove_correlation, check=check)

		train_data = full_data.loc[train_data.index,:]
		test_data = full_data.loc[test_data.index,:]

		print train_data.shape,test_data.shape

		if check:
			train_data.to_csv('explore/train_data_for_explore.csv')
			test_data.to_csv('explore/test_data_for_explore.csv')

		return train_data,train_score,test_data


if __name__ == '__main__':

	import cPickle
	import os

	try:
		os.makedirs('explore')
	except OSError:
		pass

	# 全局变量设置，以保证线上线下参数的一致
	DROP_THRESHOLD = 0.95
	REDUNDANCY_PROCESS = True
	BY_CATEGORY = True
	ZERO_EQUAL_NA = False
	AUGMENT = False
	REMOVE_CORRELATION = False
	CHECK = False

	# 训练数据的读入以及异常处理
	data = pd.read_csv('train_data.csv', index_col=0, header=0)
	data['temp_id'] = data.index
	data = data.drop_duplicates(['temp_id']).drop(labels=['temp_id'], axis=1)
	data.loc['ID452', '210X27'] = 0.004

	# 线下测试
	train_data, train_score, test_data, test_score = data_split(data, mode='offline', DROP_THRESHOLD=DROP_THRESHOLD,
	                                                            redundancy_process=REDUNDANCY_PROCESS,
	                                                            by_category=BY_CATEGORY, zero_equal_na=ZERO_EQUAL_NA,
	                                                            augment=AUGMENT, remove_correlation=REMOVE_CORRELATION,
	                                                            check=CHECK)
	cPickle.dump((train_data, train_score, test_data, test_score), open('offline_data.pkl', 'w'))
	'''
	# 线上测试
	train_data, train_score, test_data = data_split(data, mode='online', DROP_THRESHOLD=DROP_THRESHOLD,
	                                                redundancy_process=REDUNDANCY_PROCESS,
	                                                by_category=BY_CATEGORY, zero_equal_na=ZERO_EQUAL_NA,
	                                                augment=AUGMENT, remove_correlation=REMOVE_CORRELATION,
	                                                check=CHECK)
	import cPickle
	cPickle.dump((train_data, train_score, test_data), open('online_data.pkl', 'w'))
	'''
