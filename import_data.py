from odps import ODPS
import pandas as pd
import re


def __identify_categorical_variable(df):
	tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
	categorical_columns = filter(lambda x: re.match(tool_mark, x), df.columns)
	return categorical_columns



def sql_create_table(data,table_name):
	odps = ODPS('LTAIYQbRZMzJSs1V', 'DUHOB76E6mK4mm14o3NH2fD0r7im7y', 'YGTQ',
	            endpoint='http://service.odps.aliyun.com/api')
	project = odps.get_project()

	if 'Chamber ID' in data.columns:	data.rename(columns={'Chamber ID':'Chamber_ID'},inplace= True)
	categorical_columns = __identify_categorical_variable(data)
	if 'Chamber_ID' in categorical_columns:	categorical_columns.remove('Chamber_ID')
	if 'Value' in categorical_columns: categorical_columns.remove('Value')

	import re
	j = 1
	t = 1
	for i in range(len(categorical_columns)):

		if re.match('ERROR',categorical_columns[i]):
			data.rename(columns={categorical_columns[i]:'B_'+str(j)}, inplace = True)
			categorical_columns[i] = 'B_'+str(j)
			j = j+1

		if re.match('TOOL',categorical_columns[i]) or re.match('Tool',categorical_columns[i]):
			data.rename(columns={categorical_columns[i]:'TOOL_'+str(t)}, inplace = True)
			categorical_columns[i] = 'TOOL_'+str(t)
			t = t+1

	col_df  =  pd.DataFrame({'col_name':data.columns.values.tolist()})

	for i in col_df.index:
		if col_df.loc[i,'col_name'] in categorical_columns:
			col_df.loc[i,'type'] = 'string'
		else:
			col_df.loc[i,'type'] = 'double'
			col_df.loc[i,'col_name'] = 'A_' + col_df.loc[i,'col_name']

	string = ''
	for i in range(col_df.shape[0]-1):
		#string = string + col_df.loc[i,'col_name'] + ' ' + col_df.loc[i,'type'] + ', '
		string = string + col_df.loc[i,'col_name'] + ' ' + 'double' + ', '

	#string = string + col_df.loc[col_df.shape[0]-1,'col_name'] + ' ' +col_df.loc[col_df.shape[0]-1,'type']
	string = string + col_df.loc[col_df.shape[0]-1,'col_name'] + ' ' + 'double'

	odps.delete_table(table_name, if_exists=True)
	table = odps.create_table(table_name, string, if_not_exists=True)
	records = []
	for i in range(data.shape[0]):
		records.append(data.iloc[i, :].tolist())

	odps.write_table(table_name, records)

def upload_table(data,table_name, max_dim=1200):
	a = data.shape[0] / max_dim + 1
	for i  in range(a-1):
		sql_create_table(data.iloc[i*max_dim:(i+1)*max_dim,],table_name+str(i+1))
	i = a-1
	sql_create_table(data.iloc[i * max_dim:,], table_name + str(i + 1))



if __name__ == '__main__':
	import cPickle
	train_data, train_score, test_data = cPickle.load(open('online_data_upload.pkl'))
	train_data = pd.concat([train_data,train_score],axis=1)

	print(train_data.shape)
	upload_table(train_data,'train_data_')


