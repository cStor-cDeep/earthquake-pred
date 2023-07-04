import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv("DATA_TEST.csv",sep=',',encoding='utf-8')
#print(data)

print(data.shape[0])
data_train = data.iloc[:,1:]
#print(data_train)
min_max_scaler = preprocessing.MinMaxScaler()
data_train = min_max_scaler.fit_transform(data_train)
print(data_train)
data_label = data.iloc[:,0]
print(data_label)
'''
x_train, x_test, y_train, y_test = train_test_split(data_train, data_label, test_size=0.2)
print(x_train)

data_shape= data_train.values.reshape((data.shape[0], data.shape[1]-1,1,1))
print(data_shape)
'''
#train_images = data[:,].reshape((data.shape[0], data.shape[1], 1))
#print(train_images)
