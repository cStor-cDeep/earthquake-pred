import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import time

data_ga_19 = pd.read_csv("GA19.csv",sep=",",encoding="utf-8")
data_ga_19['data_time']=pd.to_datetime(data_ga_19['TimeStamp'],unit='s')
print(data_ga_19)


y1 = data_ga_19['sound@abs_max_top5p']
y2 = data_ga_19['sound@abs_max_top10p']


x = data_ga_19['data_time']


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


fig, ax = plt.subplots(figsize=(20,10))
plt.plot(x, y1, 'b',linestyle='--',marker='p',markersize=1.0,color='b',linewidth=1,label="abs_max_top5p")
plt.plot(x, y2, 'b',marker='.',markersize=1.0,color ='r',linewidth=1,label="abs_max_top10p")

plt.legend(loc="best")
plt.xticks(size=12)

plt.title('Sound@abs_max_topx',fontdict=font)
plt.xlabel("Timestamp",fontdict=font)
plt.ylabel("Value",fontdict=font)
#plt.savefig('GA_abs_max.jpg',dpi=1000)
plt.show()
