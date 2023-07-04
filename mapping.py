import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("data_19_201711_20201130.csv",sep=',',encoding = "utf-8")
#print(data.describe())
data1=data.iloc[:,2:]
#print(data1)
min_max_scaler = preprocessing.MinMaxScaler()
data_train = min_max_scaler.fit_transform(data1)
print(data_train)
#data_train = [i * 255 for i in data_train]
data2 = data_train * 255
print(data2)

'''
y_min = 0
y_max =255
data2 = (y_max-y_min)*(data1-np.min(data1))/(np.max(data1)-np.min(data1))
print(data2)
'''
#print(data2.describe())
data2.to_csv("map_data.csv",sep=",",encoding="utf-8")