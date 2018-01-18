# coding=utf-8
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def _identify_categorical_variable(df):
	tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
	categorical_columns = filter(lambda x: re.match(tool_mark, str(x)), df.columns)
	return categorical_columns

def _category_normalize(df, category, X_scaler_dict, fit=True, limit=None):
	assert len(set(df.loc[:,category])) == 1
	label = df.loc[:, category][0]
	df = df.drop(labels=category, axis=1)
	if fit:
		X_scaler_dict[category][label].fit(df)
	try:
		final_df = pd.DataFrame(X_scaler_dict[category][label].transform(df), columns=df.columns, index=df.index)
	# 出现scaler_dict中没有的类别的处理
	except KeyError:
		print '%s of %s in test data instead of train data.' % (label, category)
		X_scaler_dict[category][label] = StandardScaler()
		final_df = pd.DataFrame(X_scaler_dict[category][label].fit_transform(df), columns=df.columns, index=df.index)
	# limit参数存在时，重复极端值处理以及重新标准化的过程，直到不再有极端值
	if limit is not None:
		while (final_df > limit).sum().sum() + (final_df < -limit).sum().sum() !=0:
			final_df = final_df.apply(
				lambda x: x.apply(lambda y: x[x <= limit].max() if y > limit else y), axis=0)
			final_df = final_df.apply(
				lambda x: x.apply(lambda y: x[x >= -limit].min() if y < -limit else y), axis=0)
			if fit:
				df = pd.DataFrame(X_scaler_dict[category][label].inverse_transform(final_df), columns=df.columns,
				                  index=df.index)
				final_df = pd.DataFrame(X_scaler_dict[category][label].fit_transform(df), columns=df.columns,
				                        index=df.index)
	return final_df


def _find_best_bin(column, threshold):
	all_value = sorted(list(set(column)))
	derivative = map(lambda i: (all_value[i+1]-all_value[i])/float(all_value[i]-all_value[i-1]), xrange(1,len(all_value)-1))
	split_index = filter(lambda i: derivative[i]>threshold, xrange(len(derivative)))
	if len(split_index) == 0:
		return None
	else:
		return map(lambda i: (all_value[i+2]+all_value[i+1])/float(2),split_index)

def _all_binarize(df, bin=4):
	def __binarize_one(col):
		bin_df = pd.cut(col, bin)
		mean_df = col.groupby(by=bin_df).mean()
		bin_dict = dict(mean_df)
		return bin_df.apply(lambda x: bin_dict[x])
	final_df = df.apply(lambda x: __binarize_one(x))
	return final_df


class YGTQ_Scaler():
	# method参数控制标准化方法，max_z_score是连续变量的极端值阈值，discrete_col是离散变量的列名，extreme_process控制极端值的处理方法，discrete_max_z_score是离散变量的极端值阈值，discrete_weight是离散变量的额外权重
	def __init__(self, method='all', max_z_score=20, pca_n_components=None, discrete_col=[], extreme_process='shrink',
	             discrete_max_z_score=4, discrete_weight=1):
		self.method = method
		self.X_scaler = None
		self.y_scaler = None
		self.fit = False
		self.pca_scaler = None
		self.pca_n_components = pca_n_components
		self.limit = max_z_score
		self.discrete_col = discrete_col
		self.chosen_col = None
		assert extreme_process in ['shrink','drop_all','drop_test','drop_train']
		self.extreme_method = extreme_process
		self.discrete_max_z_score = discrete_max_z_score
		self.disrete_weight = discrete_weight

	def fit_transform(self, data, score, auxiliary_data=None, test_data=None):
		assert data.isnull().sum().sum() == 0
		assert data.shape[0] == len(score)
		if auxiliary_data is not None:
			assert auxiliary_data.isnull().sum().sum() == 0
			assert auxiliary_data.shape[1] == data.shape[1]
		self.fit = True

		# 辅助数据和训练数据合并，y_scaler的训练
		combined_data = data if auxiliary_data is None else pd.concat([data, auxiliary_data], axis=0)
		self.y_scaler = StandardScaler(with_std=False)
		score = pd.Series(self.y_scaler.fit_transform(score), index=score.index)

		# 离散变量单独做标准化，并从数据中分离
		if len(self.discrete_col)>0:
			self.discrete_col_scaler = StandardScaler()
			discrete_part = combined_data.loc[:,self.discrete_col]
			discrete_part = pd.DataFrame(self.discrete_col_scaler.fit_transform(discrete_part), columns=discrete_part.columns,
			                             index=discrete_part.index)
			discrete_part.to_csv('discrete_data.csv')
			discrete_part = discrete_part * self.disrete_weight
			discrete_part = discrete_part.apply(lambda x: x.apply(lambda y: self.discrete_max_z_score if y > self.discrete_max_z_score else y),
				axis=0)
			discrete_part = discrete_part.apply(lambda x: x.apply(lambda y: -self.discrete_max_z_score if y < -self.discrete_max_z_score else y),
			                                    axis=0)

			combined_data = combined_data.drop(labels=self.discrete_col, axis=1)
		else:
			discrete_part = pd.DataFrame()

		# 识别工具变量
		categorical_columns = _identify_categorical_variable(combined_data)

		if self.method == 'all':
			# 无视工具变量的取值，对所有数据进行标准化
			combined_data = combined_data.drop(labels=categorical_columns, axis=1)
			self.X_scaler = StandardScaler()
			if combined_data.shape[1]>0:
				combined_data = pd.DataFrame(self.X_scaler.fit_transform(combined_data), columns=combined_data.columns, index=combined_data.index)
			if self.pca_n_components is not None:
				self.pca_scaler = PCA(n_components=self.pca_n_components)
				combined_data = pd.DataFrame(self.pca_scaler.fit_transform(combined_data), index=combined_data.index)

		elif self.method == 'categorical':

			# 根据工具变量的取值，对该取值下的数据进行标准化
			import categorical_processing

			final_df = pd.DataFrame(index=combined_data.index)
			self.X_scaler = {}
			if self.pca_n_components is not None:
				self.pca_scaler = {}
			feature_dict = categorical_processing.feature_subgrouping(combined_data, categorical_columns)
			for category in categorical_columns:
				self.X_scaler[category] = {}
				for label in set(combined_data.loc[:, category]):
					self.X_scaler[category][label] = StandardScaler()
				partial_df = categorical_processing.chunk_dataframe_generator(combined_data, feature_dict, category)
				partial_df = partial_df.groupby(category).apply(
					lambda x: _category_normalize(x, category, self.X_scaler, limit=None))
				if self.pca_n_components is not None:
					self.pca_scaler[category] = PCA(n_components=min(self.pca_n_components,partial_df.shape[1]))
					partial_df = pd.DataFrame(self.pca_scaler[category].fit_transform(partial_df), index=partial_df.index,
					                          columns=map(lambda x: category+'_'+str(x), xrange(self.pca_scaler[category].n_components_)))
				final_df = pd.concat([final_df, partial_df], axis=1)
			combined_data = final_df

		else:
			print "Warning: using data without scaling!"
			combined_data = combined_data.drop(labels=categorical_columns, axis=1)

		# 某些特征在工具变量的一个取值下没有方差，会导致结果的不稳定性
		self.chosen_col = combined_data.columns[(abs(combined_data.mean())<0.01)]
		#self.chosen_col = combined_data.columns
		print len(self.chosen_col)
		combined_data = combined_data.loc[:,self.chosen_col]

		if self.extreme_method == 'drop_all':
			# 训练集和测试集上任何有异常值的维度去除
			assert test_data is not None
			chosen_col = pd.concat([combined_data,self.transform(test_data)],axis=0).apply(
				lambda x: (x >= float(self.limit)).sum() + (x <= -float(self.limit)).sum(), axis=0) == 0
			self.chosen_col = combined_data.columns[chosen_col]
		elif self.extreme_method == 'drop_test':
			# 测试集上有异常值的维度去除
			assert test_data is not None
			self.transform(test_data).apply(lambda x: (x >= float(self.limit)).sum() + (x <= -float(self.limit)).sum(),
			                    axis=1).to_csv(
				'explore/extreme_value_by_index_test.csv')
			chosen_col = self.transform(test_data)[self.chosen_col].apply(
				lambda x: (x >= float(self.limit)).sum() + (x <= -float(self.limit)).sum(), axis=0) == 0
			self.chosen_col = combined_data.columns[chosen_col]
		elif self.extreme_method == 'drop_train':
			# 训练集上有异常值的维度去除
			chosen_col = combined_data.loc[data.index, :].apply(
				lambda x: (x >= float(self.limit)).sum() + (x <= -float(self.limit)).sum(), axis=0) == 0
			self.chosen_col = combined_data.columns[chosen_col]

		# 记录一些中间过程
		combined_data.apply(lambda x: (x >= float(self.limit)).sum() + (x <= -float(self.limit)).sum(), axis=0).to_csv(
			'explore/extreme_value_by_column.csv')
		combined_data.apply(lambda x: (x >= float(self.limit)).sum() + (x <= -float(self.limit)).sum(), axis=1).to_csv(
			'explore/extreme_value_by_index.csv')
		combined_data.to_csv('explore/all_data_explore.csv')

		# 确定最终的特征矩阵，离散变量的部分放回，分离出训练数据
		combined_data = combined_data.loc[:, self.chosen_col]
		combined_data = combined_data.apply(lambda x: x.apply(lambda y: float(self.limit) if y > float(self.limit) else y), axis=0)
		combined_data = combined_data.apply(lambda x: x.apply(lambda y: -float(self.limit) if y < -float(self.limit) else y), axis=0)
		combined_data = pd.concat([combined_data, discrete_part], axis=1)

		data = combined_data.loc[data.index,]

		return data, score

	def transform(self, data):

		assert self.fit
		assert data.isnull().sum().sum() == 0

		if len(self.discrete_col)>0:
			discrete_part = data.loc[:,self.discrete_col]
			discrete_part = pd.DataFrame(self.discrete_col_scaler.transform(discrete_part),
			                             columns=discrete_part.columns,
			                             index=discrete_part.index)
			discrete_part = discrete_part * self.disrete_weight
			discrete_part = discrete_part.apply(lambda x: x.apply(lambda y: self.discrete_max_z_score if y > self.discrete_max_z_score else y),
				axis=0)
			discrete_part = discrete_part.apply(lambda x: x.apply(lambda y: -self.discrete_max_z_score if y < -self.discrete_max_z_score else y),
				axis=0)
			data = data.drop(labels=self.discrete_col, axis=1)
		else:
			discrete_part = pd.DataFrame()

		categorical_columns = _identify_categorical_variable(data)

		if self.method == 'all':
			data = data.drop(labels=categorical_columns, axis=1)
			if data.shape[1]>0:
				data = pd.DataFrame(self.X_scaler.transform(data), columns=data.columns, index=data.index)
			if self.pca_scaler is not None:
				data = pd.DataFrame(self.pca_scaler.transform(data),index=data.index)

		elif self.method == 'categorical':
			import categorical_processing
			feature_dict = categorical_processing.feature_subgrouping(data, categorical_columns)
			final_df = pd.DataFrame(index=data.index)
			for category in categorical_columns:
				partial_df = categorical_processing.chunk_dataframe_generator(data, feature_dict, category)
				partial_df = partial_df.groupby(category).apply(
					lambda x: _category_normalize(x, category, self.X_scaler, fit=False))
				if self.pca_n_components is not None:
					partial_df = pd.DataFrame(self.pca_scaler[category].transform(partial_df), index=partial_df.index,
					                          columns=map(lambda x: category+'_'+str(x), xrange(self.pca_scaler[category].n_components_)))
				final_df = pd.concat([final_df, partial_df], axis=1)
			data = final_df

		else:
			print "Warning: no scaling applied!"
			data = data.drop(labels=categorical_columns, axis=1)

		data = data.loc[:,self.chosen_col]

		data = data.apply(lambda x: x.apply(lambda y: self.limit if y > self.limit else y), axis=0)
		data = data.apply(lambda x: x.apply(lambda y: -self.limit if y < -self.limit else y), axis=0)
		data = pd.concat([data, discrete_part], axis=1)

		return data

if __name__ == '__main__':

	data = pd.read_csv('train_data.csv', index_col=0, header=0)
	data['temp_id'] = data.index
	data = data.drop_duplicates(['temp_id']).drop(labels=['temp_id'], axis=1)

	scaler = YGTQ_Scaler(method='categorical',max_z_score=4)
	transformed_data, _ = scaler.fit_transform(data.iloc[:,:30], data.Y)
	print transformed_data.shape