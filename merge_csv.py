import pandas as pd
import os
import importlib
import sys
importlib.reload(sys)
import csv
def read_csv(filepath ,sep=',',encoding='utf-8'):

    p=[]#返回列表，里面每一个都装着dataframe
    a=[]#装路径
    #获取路径文件夹下面的文件的全部路径
    for file in os.listdir(filepath):
        a.append(os.path.join(filepath, file))
    for i in range(len(a)):
        p1=[]
        path=a[i]
        p1= pd.read_csv(path,sep=',',encoding=encoding)
        print(p1.shape)
        p.append(p1)
    return p
path=r'E:\公司\地震数据\AETA地震数据\数据整理\EM_GA_2021_01_01-2021_01_31\GA_20210101-20210131'
p=read_csv(path,sep=',',encoding='utf-8')
print(p)
c=pd.concat(p)#将p里面全部dataframe合成一个
print(c)
print(c.shape)
c.to_csv("test_GA.csv",sep=',')
#with open("test1.csv", 'a', encoding='utf-8') as file:
#   csv.writer(file).writerow(c)
