import pandas as pd
import numpy as np
from PIL import Image

from sklearn import preprocessing

data =pd.read_csv("time_huidu.csv",sep=',',encoding="utf-8")
min_max_scaler = preprocessing.MinMaxScaler()
data_train = min_max_scaler.fit_transform(data)
print(data_train)   #归一化后的取值
#pd.DataFrame(data_train).to_csv("归一化.csv",sep=",",encoding='utf-8')
data2 = data_train * 255
print(data2)
data2 = np.array(data2).astype(np.int32)
print(data2)           #映射0-255并取整
#pd.DataFrame(data2).to_csv("映射后数据.csv",sep=",",encoding='utf-8')
print(len(data2))


def create_data(dataset,sequence):
    data_x = []
    for i in range(sequence,len(dataset)):
        data_x.append(dataset[i-sequence:i,0:dataset.shape[1]])
    print(data_x)
    return data_x

data_x = create_data(data2,95)
#print(data_x)
#print(len(data_x))
#print("*****************")
for i in range(len(data_x)):
    print(data_x[i])
    #pd.DataFrame(data_x[i]).to_csv("{}.csv".format(i),sep=",",encoding="utf-8")
    image1 = Image.fromarray(data_x[i])
    image1 = image1.convert('L')
    #image1.save("{}.png".format(i), dpi=(300, 300))
'''
data_01 = create_data(data_train,95)
for i in range(len(data_01)):
    print(data_01[i])
    pd.DataFrame(data_01[i]).to_csv("normalization_{}.csv".format(i), sep=",", encoding="utf-8")

'''
'''
image1 = Image.fromarray(data2)
image1 = image1.convert('L')
image1.save("output_time.png",dpi=(300,300))
'''