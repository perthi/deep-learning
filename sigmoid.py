
print("Hello World !!!!")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from array import array


def data_gen( t = 0, a = 1):
    intarray = [];
    cnt = 0
    while cnt < 8000:
        cnt += 1
        t += 0.01
        intarray.append(   1/(1 + np.exp( -a*( t - 20 ))) );
    cnt = 0;
    t2 = t;
    
    while cnt < 7000:
        cnt += 1
        t += 0.01
        intarray.append(   1/(1 + np.exp( -a*((t2-t)  +20 ))));

    return intarray;   
                


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.xlabel(r'\textbf{time} (s)')
plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
plt.plot( data_gen(0, 0.5), color = 'green' )
plt.plot( data_gen(0, 1), color = 'blue' )
plt.plot( data_gen(0, 4), color = 'red' )
plt.plot( data_gen(0, 0.3), color = 'cyan' )

plt.show()

