from sklearn.model_selection import train_test_split

def data_loader(data_path,condition_path,split_valid_ratio): #split_valid_ratioにテスト用データにするデータの割合を指定する。
  X = pd.read_csv(data_path,index_col=0)
  label = pd.read_csv(condition_path,index_col=0)
  X_train,condition_train,X_test,condition_test = train_test_split(X,condition,train_size=split_valid_ratio)
  return X_train,condition_train,X_test,condition_test
