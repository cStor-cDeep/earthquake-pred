import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import time

data_em_19 = pd.read_csv("EM19.csv",sep=",",encoding="utf-8")
data_em_19['data_time']=pd.to_datetime(data_em_19['TimeStamp'],unit='s')
print(data_em_19)

y1 = data_em_19['magn@power_0_5']
y2 = data_em_19['magn@power_5_10']
y3 = data_em_19['magn@power_10_15']
y4 = data_em_19['magn@power_15_20']
y5 = data_em_19['magn@power_20_25']
y6 = data_em_19['magn@power_25_30']
y7 = data_em_19['magn@power_30_35']
y8 = data_em_19['magn@power_35_40']
y9 = data_em_19['magn@power_40_60']

x = data_em_19['data_time']

import matplotlib as mpl
sns.set(font_scale=1.5)
#修改默认设置
mpl.rcParams["font.family"] = 'Times New Roman'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 10   #字体大小
mpl.rcParams["axes.linewidth"] = 1   #轴线边框粗细（默认的太粗了）

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10,
        }
font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10,
        }

fig, ax = plt.subplots(figsize=(20,16))
plt.subplot(331)
plt.plot(x, y1,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_0_5")

plt.legend(loc='upper right',fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(332)
plt.plot(x, y2,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_5_10")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(333)
plt.plot(x, y3,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_10_15")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(334)
plt.plot(x, y4,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_15_20")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(335)
plt.plot(x, y5,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_20_25")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(336)
plt.plot(x, y6,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_25_30")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(337)
plt.plot(x, y7,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_30_35")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(338)
plt.plot(x, y8,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_35_40")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

plt.subplot(339)
plt.plot(x, y9,linestyle='--',marker='p',markersize=1.0,
         color='b',linewidth=1,label="power_40_60")

plt.legend(loc="upper right",fontsize=6)
plt.xticks(size=5)
plt.yticks(size=7)

from matplotlib.backends.backend_pdf import PdfPages
#plt.savefig('EM_power.pdf', bbox_inches='tight')
plt.savefig('EM_power.jpg',dpi=1000)
plt.show()