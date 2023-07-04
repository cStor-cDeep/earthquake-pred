import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
time_series_array = np.sin(np.linspace(-np.pi, np.pi, 400)) + np.random.rand((400))
n_steps           = 15 #number of rolling steps for the mean/std.

#Compute curves of interest:
time_series_df = pd.DataFrame(time_series_array)
print(time_series_df)
smooth_path    = time_series_df.rolling(n_steps).mean()
print(smooth_path)
path_deviation = 2 * time_series_df.rolling(n_steps).std()

under_line     = (smooth_path-path_deviation)[0]
over_line      = (smooth_path+path_deviation)[0]

#Plotting:
plt.plot(smooth_path, linewidth=2) #mean curve.
plt.fill_between(path_deviation.index, under_line, over_line, color='b', alpha=.1) #std curves.
plt.show()
'''
'''
data = pd.read_csv("loss_accuracy1.csv",sep=",",encoding="utf-8")
print(data)
plt.plot(data['accuracy'], linewidth=2) #mean curve.

#plt.fill_between(path_deviation.index, under_line, over_line, color='b', alpha=.1) #std curves.
plt.show()
'''
'''
import mpl_toolkits.axisartist as ast
from mpl_toolkits.mplot3d import Axes3D

data =np.linspace(-10,10,100)
x,y = np.meshgrid(data,data)
print(x)
print(y)


z = x**2 + y**2 - x*y
print(z)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x,y,z,rstride=4,cstride=4)
plt.title("三维图")
plt.show()

'''
'''
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

# 生成网格点坐标矩阵
n = 1000
x, y = np.meshgrid(np.linspace(-3, 3, n),
				   np.linspace(-3, 3, n))

# 根据x,y 计算当前坐标下的z高度值
z = (1-x/2 + x**5 + y**3) * np.exp(-x**2 -y**2)

mp.figure('Surface', facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('Time', fontsize=14)
ax3d.set_ylabel('Site', fontsize=14)
ax3d.set_zlabel('Var', fontsize=14)
ax3d.plot_surface(x, y, z, rstride=50,
	cstride=50, cmap='jet')
mp.show()
'''

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['r', 'g', 'b', 'y']
yticks = [3, 2, 1, 0]
for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = np.arange(20)
    print(xs)
    ys = np.random.rand(20)
    print(ys)
    # You can provide either a single color or an array with the same length as
    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'

    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)
    #ax.plot(xs, ys, zs=k, zdir='y', color=cs)
ax.set_xlabel('Character')
ax.set_ylabel('Site')
ax.set_zlabel('Time')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)

plt.show()

'''
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import math
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


ax = plt.figure().add_subplot(projection='3d')

x = np.linspace(0., 10., 31)        #生成等间距数据的一个函数
lambdas = range(1, 9)

# verts[i] is a list of (x, y) pairs defining polygon i.
gamma = np.vectorize(math.gamma)    #
verts = [polygon_under_graph(x, l**x * np.exp(-l) / gamma(x + 1))
         for l in lambdas]
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
ax.add_collection3d(poly, zs=lambdas, zdir='y')

ax.set(xlim=(0, 10), ylim=(1, 9), zlim=(0, 0.35),
       xlabel='x', ylabel=r'$\lambda$', zlabel='probability')

plt.show()
'''
'''
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
z = np.linspace(0, 300,61)



ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
'''

