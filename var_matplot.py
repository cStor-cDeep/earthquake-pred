
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['r', 'g', 'b', 'y','g'] *10
yticks = []
for i in range(50):
    yticks.append(i)
yticks.reverse()
#yticks = [3, 2, 1, 0]
for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = np.arange(95)
    ys = np.random.randint(0,150,95)
    #ys = np.random.rand(95)
    # You can provide either a single color or an array with the same length as
    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'

    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)
ax.set_xlabel('Factor')
ax.set_ylabel('Station')
ax.set_zlabel('Time')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks((0,10,20,30,40,50))

plt.show()