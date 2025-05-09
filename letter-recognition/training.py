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
from utils import *
import sys

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)

print(tf.__version__)
NO_EPOCS = 50

X, y = load_data()
m, n = X.shape

plot_random(8, 8, X, y, figsize=(8,8))

layers_n = [25, 10, 15, 1]
#layers_n = [25, 15, 1]

model = Sequential( [tf.keras.Input(shape=(400,)),], name = "my_model")

for l in layers_n:
    model.add( Dense(l, activation="sigmoid") )

model.summary()

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
    A = dense_propagation_v(X,  W[0], b[0], sigmoid)
    length = len(W) - 1

    i : int = 1
    while length !=0:
        A =  dense_propagation_v(A,  W[i], b[i], sigmoid)
        length -= 1
        i+= 1

    return A


W = []
b = []
layers =  model.layers

for l in layers:
    w_l, b_l = l.get_weights()
    W.append(w_l)
    b.append(b_l)

prediction = inference_v2(X, W, b)
#sys.exit()
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


for i in range(len(indexes)):
    print("index = {}, value = {}, yhat = {}".format(indexes[i], values[i], Yhat[ indexes[i] ] ))

print("{} out of {} images was miss classified ( {} %)".format(len_err, len_all, percent))


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
