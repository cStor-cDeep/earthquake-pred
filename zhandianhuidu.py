import pandas as pd
import numpy as np
from PIL import Image

from sklearn import preprocessing

data =pd.read_csv("zhandian_huidu.csv",sep=',',encoding="utf-8")
min_max_scaler = preprocessing.MinMaxScaler()
data_train = min_max_scaler.fit_transform(data)
data2 = data_train * 255
print(data2)
data2 = np.array(data2).astype(np.int32)
print(data2)
'''
#data3 = pd.DataFrame(data2)
#data3.to_csv("test_huidu.csv",sep=',',encoding='utf-8')
'''
image1 = Image.fromarray(data2)
image1 = image1.convert('L')
image1.save("output_zhandian.png",dpi=(300,300))