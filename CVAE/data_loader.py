from sklearn.model_selection import train_test_split
import pandas as pd

#data_path:トレーニングデータのパス
#condition_path:ラベルデータのパス
#split_valid_ratio:トレーニングデータとテストデータの分割割合。トレーニングデータの割合をfloatで
def data_loader(data_path,condition_path,split_valid_ratio):
  import pandas as pd
  X = pd.read_csv(data_path,index_col=0)
  condition = pd.read_csv(condition_path,index_col=0)
  X_train,condition_train,X_test,condition_test = train_test_split(X,condition,train_size=split_valid_ratio)
  return X_train,condition_train,X_test,condition_test
