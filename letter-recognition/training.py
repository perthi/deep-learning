from utils import *
from nn_model import *

import numpy as np
import tensorflow as tf
import logging
import warnings

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)

NO_EPOCS = 50
#layers_n = [25, 15, 1]

model = genrate_model(layers=[25,15,1], input_size= input_size)

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
