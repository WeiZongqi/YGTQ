from odps import ODPS
import pandas as pd
import re


def __identify_categorical_variable(df):
	tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
	categorical_columns = filter(lambda x: re.match(tool_mark, x), df.columns)
	return categorical_columns



def sql_create_table(odps, data, table_name):
	project = odps.get_project()

	if 'Chamber ID' in data.columns:	data.rename(columns={'Chamber ID' : 'Chamber_ID'}, inplace= True)
	categorical_columns = __identify_categorical_variable(data)
	if 'Chamber_ID' in categorical_columns: categorical_columns.remove('Chamber_ID')

	import re
	j = 1
	t = 1
	for i in range(len(categorical_columns)):

		if re.match('ERROR',categorical_columns[i]):
			data.rename(columns={categorical_columns[i]:'B_'+str(j)}, inplace = True)
			j = j+1

		if re.match('TOOL',categorical_columns[i]) or re.match('Tool',categorical_columns[i]):
			data.rename(columns={categorical_columns[i]:'TOOL_'+str(t)}, inplace = True)
			t = t+1
			

	string = ''
	for col in data.columns:
		#string = string + col_df.loc[i,'col_name'] + ' ' + col_df.loc[i,'type'] + ', '
		string = string + 'A_'+ col + ' ' + 'double' + ', '
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


	print(train_data.shape)
	upload_table(odps, train_data,'train_data_')


