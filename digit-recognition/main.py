from utils import *
from nn_model import *
from config import *

import numpy as np
import tensorflow as tf
import logging
import warnings

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)

model = generate_model(LAYERS, input_size= input_size)
X_train, y_train,  X_validate, y_validate = load_data()
model.fit(X_train,y_train,epochs=NO_EPOCS)
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

prediction_val = inference_v(X_validate, W, b)
y_predicted_val = (prediction_val >= 0.5).astype(int)

print("*** TEST STATISTICS FOR VALIDATION DATA****")
print_statistics(X_validate, y_validate, y_predicted_val, plot_misclassified= False)

prediction_train = inference_v(X_train, W, b)
y_predicted_train = (prediction_train >= 0.5).astype(int)

print("*** TEST STATISTICS FOR TRAINING DATA****")
print_statistics(X_train, y_train, y_predicted_train, plot_misclassified=False)