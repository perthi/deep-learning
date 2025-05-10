import numpy as np
import tensorflow as tf
from keras.api.layers import *
from keras.api.optimizers import *
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.losses import *
import logging
import warnings
from utils import *

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)

print(tf.__version__)

NO_EPOCS = 50
layers_n = [25, 15, 1]

model = Sequential( [tf.keras.Input(shape=(input_size,)),], name = "my_model")
#model = Sequential( [tf.keras.Input(shape=(400,)),], name = "my_model")


for l in layers_n:
    model.add( Dense(l, activation="sigmoid") )

model.summary()
model.compile( loss= BinaryCrossentropy(), optimizer= Adam(0.001),)

X, y = load_data()

model.fit(X,y,epochs=NO_EPOCS)
#plot_random_with_prediction(8, 8, X, y, model, figsize=(8,8))

def dense_propagation_v(A_in, W, b, g):
    AT = A_in
    Z = np.matmul(AT,W) + b
    A_out = g(Z)
    return(A_out)


def inference_v(X, W, b):
    A = dense_propagation_v(X,  W[0], b[0], sigmoid)
    length = len(W) - 1

    i : int = 1
    while length !=0:
        A =  dense_propagation_v(A,  W[i], b[i], sigmoid)
        length -= 1
        i+= 1
    return A


def get_parameters(layers):
    W = []
    b = []
    for l in layers:
        w_l, b_l = l.get_weights()
        W.append(w_l)
        b.append(b_l)
    return W, b

W, b = get_parameters( model.layers )

prediction = inference_v(X, W, b)
Yhat = (prediction >= 0.5).astype(int)

#plot_random_with_prediction_v(8, 8, X, y, Yhat, figsize=(8,8))
errors = np.where(y != Yhat)
print_statistics(errors, X, y, Yhat)
