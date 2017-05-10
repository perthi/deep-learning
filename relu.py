

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


rc('text', usetex=True)



t1 = np.arange(-10, 10.0, 0.02)



ax1=plt.subplot(311)
plt.ylabel(r"$\frac{1}{1 + e^{-x}}$");
plt.setp(ax1.get_xticklabels(), visible=False)
plt.plot(t1, np.tanh(t1), 'b-')


ax2=plt.subplot(312, sharex=ax1)
plt.ylabel(r"$\tanh{x} $")
plt.setp(ax2.get_xticklabels(), visible=False)
plt.plot(t1, 1/(1 + np.exp(-t1)), 'b-')


ax3=plt.subplot(313, sharex=ax1)
plt.ylabel( r"$ max(0,x)$" );
plt.xlabel(r"$x$");
plt.setp(ax3.get_xticklabels(), visible=True)
plt.plot(t1, np.maximum(0,t1), 'b-')

##plt.title("blahhhhhhhhh")
plt.savefig("activation_functions.pdf");

plt.show()
