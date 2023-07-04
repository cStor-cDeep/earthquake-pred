import pandas as pd
import numpy as np
#读取震级数据
data_timestamp = pd.read_csv(r'E:\公司\地震数据\AETA地震数据\数据整理\地震目录2016_10_01-2020_11_30\EC_2016_10_01-2020_11_30.csv',
                             sep=',', encoding = "utf-8")

time_magnitude = data_timestamp[['Timestamp','Magnitude']]
#读取合并后的地磁和地声数据
data_EG = pd.read_csv("data_172_201711_20201130.csv",sep=',',encoding="utf-8")
#filenames=data_EG.columns

data_all = pd.DataFrame()
for i in range(len(time_magnitude)):
    time =int(time_magnitude['Timestamp'][i])    #获取地震时间
    magnitude = time_magnitude['Magnitude'][i]    #获取地震震级
    #print(time)
    #print(magnitude)
    data_EG.loc[data_EG.TimeStamp.between(time, time + 6000), 'Magnitude'] = magnitude
    data = data_EG[data_EG.TimeStamp.between(time,time+6000)]     #地震时间前一百分钟
    data_all = data_all.append(data)

print(data_all)
data_all.to_csv("data_all.csv",sep=',',mode='a+',encoding="utf-8")


'''
with open(r'E:\公司\地震数据\AETA地震数据\数据整理\EM_GA_2021_01_01-2021_01_31\集成.csv','a+',encoding='utf8') as f:
    a=csv.writer(f)
    a.writerow(data_all)
'''
