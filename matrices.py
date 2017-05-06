
import numpy as np
from scipy import linalg

from scipy import random, linalg, dot, diag, all, allclose
a = np.array([[1, 2], [2, 1]])
#print (a)

#b = np.linalg.inv(a);
#c = np.dot(a, b);
#print(c);
# AA, BB, Q, Z = linalg.qz(a, a)

Q, R = linalg.qr(a);

print("Q = \n", Q);
print("R = \n", R);

print("QR = \n", np.dot(Q, R) );

Q2, R2 =  linalg.rq(Q);

print("Q2 = \n", Q2);
print("R2 = \n", R2);
print("Q2R = \n", np.dot( Q2, R2 ) );

Q4 = np.dot(Q2, R2);

print(Q4);
