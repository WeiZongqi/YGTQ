from odps import ODPS
import pandas as pd
import re


def __identify_categorical_variable(df):
	tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
	categorical_columns = filter(lambda x: re.match(tool_mark, x), df.columns)
	return categorical_columns


def column_mapping(data):

	categorical_columns = __identify_categorical_variable(data)
	try:
		categorical_columns.remove('Value')
	except:
		pass

	j = 1
	t = 1
	mapping_dict = {}

	for i in range(len(categorical_columns)):
		if re.match('ERROR', categorical_columns[i]) or re.match('Chamber',categorical_columns[i]):
			data.rename(columns={categorical_columns[i]: 'B_' + str(j)}, inplace=True)
			mapping_dict[categorical_columns[i]] = 'B_' + str(j)
			j = j + 1
		else:
			data.rename(columns={categorical_columns[i]: 'TOOL_' + str(t)}, inplace=True)
			mapping_dict[categorical_columns[i]] = 'TOOL_' + str(t)
			t = t + 1

	data.columns = map(lambda x: 'A_'+x.replace(" ","") if re.match(r'\d.*',x) else x, data.columns)

	return data

def sql_create_table(odps, data, table_name):

	project = odps.get_project()

	string = ''
	for col in data.columns:
		#string = string + col_df.loc[i,'col_name'] + ' ' + col_df.loc[i,'type'] + ', '
		assert ' ' not in col
		string = string + col + ' ' + 'double' + ', '
	string = string[:-2]

	odps.delete_table(table_name, if_exists=True)
	table = odps.create_table(table_name, string, if_not_exists=True)
	records = []
	for i in range(data.shape[0]):
		records.append(data.iloc[i, :].tolist())

	odps.write_table(table_name, records)

def upload_table(odps, data, table_name, max_dim=1200):
	a = data.shape[1] / max_dim + 1
	for i  in range(a-1):
		sql_create_table(odps, data.iloc[:,i*max_dim:(i+1)*max_dim],table_name+str(i+1))
	i = a-1
	sql_create_table(odps, data.iloc[:,i * max_dim:], table_name + str(i + 1))



if __name__ == '__main__':
	import cPickle
	train_data, train_score, test_data = cPickle.load(open('online_data_upload.pkl'))
	train_data = pd.concat([train_data,train_score],axis=1)

	odps = ODPS('LTAIYQbRZMzJSs1V', 'DUHOB76E6mK4mm14o3NH2fD0r7im7y', 'YGTQ',
	            endpoint='http://service.odps.aliyun.com/api')

	print train_data.shape
	train_data = column_mapping(train_data)
	upload_table(odps, train_data,'train_data_')

	print test_data.shape
	test_data = column_mapping(test_data)
	assert  (test_data.columns != train_data.columns[:-1]).sum() == 0
	upload_table(odps,test_data,'test_data_')


