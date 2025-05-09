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
import sys

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)

print(tf.__version__)
NO_EPOCS = 30

X, y = load_data()
m, n = X.shape

plot_random(8, 8, X, y, figsize=(8,8))

layers_n = [25, 10, 15, 1]
model = Sequential( [tf.keras.Input(shape=(400,)),], name = "my_model")

for l in layers_n:
    model.add( Dense(l, activation="sigmoid") )

model.summary()

"""
[layer1, layer2, layer3, layer4] = model.layers
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
W4,b4 = layer4.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
print(f"W4 shape = {W4.shape}, b4 shape = {b4.shape}")
"""
#print(model.layers[2].weights)
    
#sys.exit()
model.compile( loss= BinaryCrossentropy(), optimizer= Adam(0.001),)
model.fit(X,y,epochs=NO_EPOCS)

prediction_debug(X, model)

fig = plot_random_with_prediction(8, 8, X, y, model, figsize=(8,8))
fig.suptitle("Label, yhat", fontsize=16)


def dense_propagation_v(A_in, W, b, g):
    AT = A_in
    Z = np.matmul(AT,W) + b
    A_out = g(Z)
    return(A_out)

    
def inference_v2(X, W, b):
    A1 = dense_propagation_v(X,  W[0], b[0], sigmoid)
    A2 = dense_propagation_v(A1, W[1], b[1], sigmoid)
    A3 = dense_propagation_v(A2, W[2], b[2], sigmoid)
    print("A3 TYPE = ", type(A3))
    A4 = dense_propagation_v(A3, W[3], b[3], sigmoid)
    return(A4)

def inference_v3(X, W, b):
    A1 = dense_propagation_v(X,  W[0], b[0], sigmoid)
    A2 = dense_propagation_v(A1, W[1], b[1], sigmoid)
    A3 = dense_propagation_v(A2, W[2], b[2], sigmoid)
    A4 = dense_propagation_v(A3, W[3], b[3], sigmoid)
    return(A4)


W = []
b = []
layers =  model.layers

for l in layers:
    w_l, b_l = l.get_weights()
    W.append(w_l)
    b.append(b_l)

prediction = inference_v2(X, W, b)

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

plt.show()
