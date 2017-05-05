
import sys
print (sys.argv[1:] )
import pandas as pd
import numpy as np

df = pd.read_csv('ML_Data_Insight_121016.csv', header=1)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

x1 = df.ix[1:, 'x1']
x2 = df.ix[1:, 'x2']
x3 = df.ix[1:, 'x3']
y =  df.ix[1:, 'y']


if sys.argv[1:] == ['winter']:
    p=ax2.scatter(x1, x2, x3, c=y, cmap=plt.winter())
elif sys.argv[1:] == ['cool']:
    p=ax2.scatter(x1, x2, x3, c=y, cmap=plt.cool())
else:
    p=ax2.scatter(x1, x2, x3, c=y, cmap=plt.hot())


fig.colorbar(p)
    
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('X3')

plt.show();

maxy=int(round(max(y)))
m = np.zeros((1,maxy))

for i in range(maxy):
    m[0,i] = i/maxy

img=plt.imshow(m, cmap=plt.cool(), aspect=2 )
plt.colorbar(img, ax=ax2)

plt.show()  




