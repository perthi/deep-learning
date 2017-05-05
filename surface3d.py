
import pandas as pd

df = pd.read_csv('ML_Data_Insight_121016.csv', header=1)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = df.ix[1:, 'x1']
x2 = df.ix[1:, 'x2']
x3 = df.ix[1:, 'x3']
y =  df.ix[1:, 'y']

ax.scatter(x1, x2, x3, c=y, cmap=plt.hot())
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

##plt.colorbar(ax)

plt.show()


