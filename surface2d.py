
import pandas as pd
import sys
print (sys.argv[1:] )
df = pd.read_csv('ML_Data_Insight_121016.csv', header=1)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    
zs = df.ix[1:, 'y']
xs = df.ix[1:, 'x2']
ys = df.ix[1:, 'x3']

#plt.show()

ax2.scatter(xs, ys, zs, c=c, marker=m)

ax2.set_xlabel('x2')
ax2.set_ylabel('x3')
ax2.set_zlabel('y')

plt.show()



