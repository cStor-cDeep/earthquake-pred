import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import time
from pylab import *

data_em_19 = pd.read_csv("EM19.csv",sep=",",encoding="utf-8")
data_em_19['data_time']=pd.to_datetime(data_em_19['TimeStamp'],unit='s')
print(data_em_19)
data_em_19.to_csv("em_19.csv",encoding="gbk")
data_em_101 = pd.read_csv("EM101.csv",sep=",",encoding="utf-8")

y1 = data_em_19['magn@abs_max_top5p']
y2 = data_em_19['magn@abs_max_top10p']
y3 = data_em_19['magn@ulf_abs_max_top5p']
y4 = data_em_19['magn@ulf_abs_max_top10p']

x = data_em_19['data_time']
#for i in rang(len(x)):
#    data_time = time.localtime(x[i])
#    data_time = time.strftime("%Y-%m-%d %H:%M:%S", data_time)


import matplotlib as mpl
sns.set(font_scale=1.5)
#修改默认设置
mpl.rcParams["font.family"] = 'Times New Roman'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 10   #字体大小
mpl.rcParams["axes.linewidth"] = 1   #轴线边框粗细（默认的太粗了）

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 15,
        }

fig, ax = plt.subplots(figsize=(20,18))
plt.subplot(211)

plt.plot(x, y1, 'b',linestyle='--',marker='p',markersize=1.0,color='b',linewidth=1,label="abs_max_top5p")
plt.plot(x, y2, 'b',marker='.',markersize=1.0,color ='r',linewidth=1,label="abs_max_top10p")

plt.legend(loc="best")
plt.xticks(size=12)

plt.title('Magn@abs_max_topx',fontdict=font1)
#plt.xlabel("timestamp",fontdict=font)
plt.ylabel("Value",fontdict=font)
#plt.xlabel("Timestamp",fontdict=font)

plt.subplot(212)
plt.plot(x, y3, 'b',linestyle='--',marker='p',markersize=1.0,color='b',linewidth=1,label="ulf_abs_max_top5p")
plt.plot(x, y4, 'b',marker='.',markersize=1.0,color ='r',linewidth=1,label="ulf_abs_max_top10p")

plt.legend(loc="best")
plt.xticks(size=12)


subplots_adjust(hspace = 0.3)
plt.title('Magn@ulf_abs_max_topx',fontdict=font1)
plt.xlabel("Timestamp",fontdict=font)
plt.ylabel("Value",fontdict=font)
plt.show()
plt.savefig('Magn@abs_max_topx.jpg',dpi=500)
'''
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(x, y1, 'b',linestyle='--',marker='p',markersize=1.0,color='b',linewidth=1,label="abs_max_top5p")
plt.plot(x, y2, 'b',marker='.',markersize=1.0,color ='r',linewidth=1,label="abs_max_top10p")

plt.legend(loc="best")
plt.xticks(size=12)

plt.title('magn@abs_max_topx',fontdict=font)
plt.xlabel("timestamp",fontdict=font)
plt.ylabel("value",fontdict=font)

plt.show()
'''