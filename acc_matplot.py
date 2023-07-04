'''
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("loss_accuracy1.csv",sep=",",encoding="utf-8")
print(data)

plt.plot(data['accuracy1'], linewidth=2)
plt.show()

'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
sns.set(font_scale=1.5)
#修改默认设置
mpl.rcParams["font.family"] = 'Times New Roman'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 10   #字体大小
mpl.rcParams["axes.linewidth"] = 1   #轴线边框粗细（默认的太粗了）
font = {'family' : 'Times New Roman',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 25,
        }

data = pd.read_csv("loss_accuracy_500epochs.csv",sep=",",encoding="utf-8")
print(data)
fig, ax = plt.subplots(figsize=(14,10))

x=data['epoch'][300:500]
y=data['accuracy_1'][300:500]
y_lower = data['accuracy_lower_1'][300:500]
y_upper = data['accuracy_upper_1'][300:500]

#mean=data['mean']


plt.plot(x, y, 'b',linestyle='--',marker='p',markersize=1.0,color='b',linewidth=1,label="accuracy")
plt.plot(x, y_lower, 'b',marker='.',markersize=1.0,color ='r',linewidth=1,label="lower")
plt.plot(x, y_upper, 'b',marker='o',markersize=1.0,color='r',linewidth=1,label="upper")
plt.fill_between(x,y,y_lower,color='red',alpha=0.1)
plt.fill_between(x,y,y_upper,color='red',alpha=0.1)

plt.legend(loc="best")
#plt.plot(x, mean, 'k',marker='o',markersize=8, linewidth=0.5)
plt.title('Model accuracy',fontdict=font)
plt.xlabel("Epoch",fontdict=font)
plt.ylabel("Accuracy",fontdict=font)
plt.savefig('Acc.jpg',dpi=1000)
plt.show()