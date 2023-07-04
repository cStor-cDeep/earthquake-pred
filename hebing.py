import pandas as pd

data1 = pd.read_csv(r"E:\公司\地震数据\AETA地震数据\数据整理\地声\sound\172_sound.csv",sep=",",encoding = 'utf-8')
data2 = pd.read_csv(r"E:\公司\地震数据\AETA地震数据\数据整理\地磁\地磁\172_magn.csv",sep=',',encoding = "utf-8")

data = pd.merge(data1, data2, on=["StationID","TimeStamp"])

#data = data[data.TimeStamp.between(1611366150,1611367150)]
print(data)
print(data.shape)
data.to_csv("data_172_201711_20201130.csv",sep=",",encoding="utf-8")

