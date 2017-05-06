import numpy as np
import matplotlib.pyplot as plt
#mat = '''SOME MATRIX'''
#plt.imshow(mat, origin="lower", cmap='gray', interpolation='nearest')
#plt.show()
m = np.zeros((1,40))
print(m)
for i in range(40):
    m[0,i] = (i*5)/100.0
print(m)
plt.imshow(m, cmap=plt.cool(), aspect=2)
#plt.yticks(np.arange(0))
#plt.xticks(np.arange(0,25,5), [0,25,50,75,100])
plt.show()

import pandas as pd
df = pd.read_csv('ML_Data_Insight_121016.csv', header=1)
y =  df.ix[1:, 'y']

maxy=int(round(max(y)))
m = np.zeros((1,maxy))

for i in range(maxy):
    m[0,i] = i/maxy

plt.imshow(m, cmap=plt.cool(), aspect=2)

plt.show()  
