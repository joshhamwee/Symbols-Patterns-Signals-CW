import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

counter = 0
xs = [0]
ys = [0]

def animate(i):
    global t,counter
    counter += 0.1
    xs.append(counter)
    ys.append(math.sin(counter))
    ax1.clear()
    plt.plot(xs,ys,color="blue")


ani = animation.FuncAnimation(fig,animate,interval=50)
plt.show()
