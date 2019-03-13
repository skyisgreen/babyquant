import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import *


#面向对象
x = np.linspace(0, 5, 10)
y = x ** 3
fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
    ax.set_title('sam')


show()
