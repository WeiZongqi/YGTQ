# coding=utf-8
import pandas as pd

def pca_processing(df, n_components, test_data=None,pca=None):
    from sklearn.decomposition import PCA
    if pca is None:
        pca = PCA(n_components=n_components)

        num_data_pca = pca.fit_transform(df)
        num_data_pca = pd.DataFrame(num_data_pca,index=df.index)
        print num_data_pca.shape

        df = num_data_pca
        print(df.shape)
        return df,pca

    else:
        num_data_pca = pca.transform(df)
        num_data_pca = pd.DataFrame(num_data_pca,index=df.index)
        print num_data_pca.shape
        df = num_data_pca
        print(df.shape)
        return df

def redundancy_processing(df):
    from scipy.stats import pearsonr
    def __one_step_processing(df, col_name):
        #corr = df.apply(lambda x: (x==df[col_name]).sum()/float(df.shape[0]), axis=0)
        corr = df.apply(lambda x: pearsonr(x,df[col_name])[0], axis=0)
        redundancy_col = df.columns[corr==1]
        redundancy_dict[col_name] = redundancy_col
        df = df.drop(labels=redundancy_col, axis=1)
        return df

    assert df.isnull().sum().sum() == 0
    redundancy_dict = {}
    final_df = pd.DataFrame(df.iloc[:,0],index=df.index)
    df = df.drop(labels=df.columns[0], axis=1)
    while df.shape[1]>0:
        col_name = df.columns[0]
        final_df = pd.concat([final_df, df[col_name]], axis=1)
        df = __one_step_processing(df, col_name)

    return final_df, redundancy_dict

def correlation_remove(df, threshold=0.8, method='remove'):
    assert df.isnull().sum().sum() == 0
    assert method in ['remove','average','diff', 'average_diff']

    from scipy.stats import pearsonr
    final_df = pd.DataFrame(df.iloc[:, 0], index=df.index)
    df = df.drop(labels=df.columns[0], axis=1)
    drop_column = []
    current_col = df.iloc[:,0].copy()
    temp_column = [df.columns[0]]
    for col in df.columns[1:]:
        if pearsonr(current_col, df[col])[0]>=threshold:
            temp_column.append(col)
            if method in['average','remove']:
                drop_column.append(col)
        else:
            if len(temp_column)>1:
                if method == 'average':
                    df.loc[:,temp_column[0]] = df.loc[:,temp_column].mean(axis=1)
                elif method == 'diff':
                    df.loc[:, temp_column[1:]] = df.loc[:, temp_column[1:]].apply(lambda x: x - current_col)
                elif method == 'average_diff':
                    df.loc[:, temp_column[0]] = df.loc[:, temp_column].mean(axis=1)
                    df.loc[:, temp_column[1:]] = df.loc[:, temp_column[1:]].apply(lambda x: x - current_col)
            current_col = df[col].copy()
            temp_column = [col]
    df = df.drop(labels=drop_column, axis=1)
    return pd.concat([final_df, df], axis=1)


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('train_data.csv', index_col=0, header=0)
    data['temp_id'] = data.index
    data = data.drop_duplicates(['temp_id']).drop(labels=['temp_id'], axis=1)
    final_df = correlation_remove(data.iloc[:,1:30], method='average_diff')

    import cPickle
    train_data, train_score, test_data = cPickle.load(open('online_data.pkl'))
    def __identify_categorical_variable(df):
        # 分类变量
        import re
        tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
        categorical_columns = filter(lambda x: re.match(tool_mark, x), df.columns)
        return categorical_columns
    categorical_columns = __identify_categorical_variable(train_data)
    train_data.drop(labels=categorical_columns, axis=1, inplace=True)
    test_data.drop(labels=categorical_columns, axis=1, inplace=True)


    N_components = 400
    train_data, pca = pca_processing(train_data, n_components=N_components)
    test_data = pca_processing(test_data, N_components, pca=pca)

    from sklearn.linear_model import LassoCV
    from regressor import regressor_train

    regressor = LassoCV(normalize=False, n_jobs=-1, n_alphas=300, cv=5)
    regressor, scaler, col_name = regressor_train(regressor, train_data, train_score, test_data=test_data,
                                                  normalize='all')
    print regressor.alpha_

    # Dump the model and column name
    cPickle.dump((col_name, regressor, scaler), open('Lasso_Regressor_with_PCA.pkl', 'w'))
    raise KeyboardInterrupt






'''
def pca(n_components):
    train_data, train_score, test_data, test_score = cPickle.load(open('data_offline.pkl'))
    print(train_data.shape)
    def zeroMean(dataMat):
        meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
        newData = dataMat - meanVal
        return newData, meanVal

    newData, meanVal = zeroMean(train_data)
    #covMat = np.cov(newData, rowvar=0)
    covMat = np.cov(newData.T, rowvar=1)

    #train_data = train_data - np.mean(train_data, axis=0) #减去均值
    print(covMat.shape)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #特征值特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n_components + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    u_eigVect = np.dot(train_data,n_eigVect)
    lowDDataMat = np.dot(u_eigVect.T,train_data)
    #lowDDataMat = train_data * n_eigVect  # 低维特征空间的数据
    #reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    #reconMat = np.dot(lowDDataMat,n_eigVect.T)+meanVal #这句不知道错哪里了！！！


    return lowDDataMat, reconMat

    print()
'''
