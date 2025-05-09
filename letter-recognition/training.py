import numpy as np
import tensorflow as tf
from keras.api.layers import *
from keras.api.optimizers import *
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.losses import *

import matplotlib.pyplot as plt
import matplotlib.axes._axes as Axes
import logging
import warnings
#from autils import *
from utils import *

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)

print(tf.__version__)
NO_EPOCS = 20

X, y = load_data()
m, n = X.shape

plot_random(8, 8, X, y, figsize=(8,8))

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        Dense(25, activation="sigmoid"),
        Dense(15, activation="sigmoid"),
        Dense(1, activation="sigmoid")
    ], name = "my_model" 
)

model.summary()

L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )


[layer1, layer2, layer3] = model.layers
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
print(model.layers[2].weights)

model.compile(
    loss= BinaryCrossentropy(),
    optimizer= Adam(0.001),
)

model.fit(
    X,y,
    epochs=NO_EPOCS
)

prediction_debug(X, model)

fig = plot_random_with_prediction(8, 8, X, y, model, figsize=(8,8))
fig.suptitle("Label, yhat", fontsize=16)

x = X[0].reshape(-1,1)         # column vector (400,1)
z1 = np.matmul(x.T,W1) + b1    # (1,400)(400,25) = (1,25)
a1 = sigmoid(z1)

def dense_propagation_v(A_in, W, b, g):
    AT = A_in
    Z = np.matmul(AT,W) + b
    A_out = g(Z)
    return(A_out)


X_tst = 0.1*np.arange(1,9,1).reshape(4,2) # (4 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(1,3) # (1,3 features)
A_tst = dense_propagation_v(X_tst, W_tst, b_tst, sigmoid)
print(A_tst)

def inference_v(X, W1, b1, W2, b2, W3, b3):
    A1 = dense_propagation_v(X,  W1, b1, sigmoid)
    A2 = dense_propagation_v(A1, W2, b2, sigmoid)
    A3 = dense_propagation_v(A2, W3, b3, sigmoid)
    return(A3)

W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()


prediction = inference_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )

Yhat = (prediction >= 0.5).astype(int)
print("predict a zero: ",Yhat[0], "predict a one: ", Yhat[500])

fig = plot_random_with_prediction_v(8, 8, X, y, Yhat, figsize=(8,8))
fig.suptitle("Label, Yhat", fontsize=16)

#plt.show()
fig = plt.figure(figsize=(1, 1))

errors = np.where(y != Yhat)


################################################


print("tuple[0]", errors[0])
print("tuple[1]", errors[1])

indexes = errors[0]
values =  errors[1]

len_err = len(values)
len_all = len(y)
percent = 100*(len_err/len_all)

print("{} out of {} images was miss classified ( {} %)".format(len_err, len_all, percent))


for i in range(len(indexes)):
    print("index = {}, value = {}, yhat = {}".format(indexes[i], values[i], Yhat[ indexes[i] ] ))


def plot_image(index):
    print("index = ", index )
    X_random_reshaped = X[index].reshape((20, 20)).T
    plt.imshow(X_random_reshaped, cmap='gray')
    plt.title(f"actual {y[index,0]}, predicted {Yhat[index, 0]}")
    plt.axis('off')

for i in range(len(values)):
    plot_image(indexes[i])
    plt.show()

print("LEN = ", len(indexes)  )

plt.show()
