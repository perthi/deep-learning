
import pandas as pd
import sys
print (sys.argv[1:] )
df = pd.read_csv('ML_Data_Insight_121016.csv', header=1)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


#def randrange(n, vmin, vmax):
#    '''
#    Helper function to make an array of random numbers having shape (n, )
#    with each number distributed Uniform(vmin, vmax).
#    '''
#    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

#n = 100

    
xs = df.ix[1:, sys.argv[1]]
ys = df.ix[1:, sys.argv[2]]
zs = df.ix[1:, sys.argv[3]]



ax2.scatter(xs, ys, zs, c='b', marker='*')

ax2.set_xlabel(sys.argv[1])
ax2.set_ylabel(sys.argv[2])
ax2.set_zlabel(sys.argv[3])

plt.show()



