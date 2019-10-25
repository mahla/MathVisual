import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


## z = a**2
##Real part
def fun(x, y):
    return x ** 2 - y ** 2 + 1


fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Real part of z')

##Imaginary part
def fun(x, y):
    return 2*x*y


ax2 = fig.add_subplot(1,2,2, projection='3d')
x2 = y2 = np.arange(-3.0, 3.0, 0.05)
X2, Y2 = np.meshgrid(x2, y2)
zs2 = np.array([fun(x2, y2) for x2, y2 in zip(np.ravel(X2), np.ravel(Y2))])
Z2 = zs2.reshape(X2.shape)

ax2.plot_surface(X2, Y2, Z2)

ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
ax2.set_title('Imaginary part of z')
plt.show()
