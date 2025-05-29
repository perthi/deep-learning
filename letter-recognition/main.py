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

#NO_EPOCS = 50
# model = generate_model(layers=[25, 15,1], input_size= input_size)
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

prediction = inference_v(X_validate, W, b)
y_predicted = (prediction >= 0.5).astype(int)

for x in y_predicted:
    print("x = {}, argmax(x)".format(x),np.argmax(x))

for z in y_validate:
    print(z)

indexes = []
values  = []

for i in range(len(y_validate)):
    argmax_val  = np.argmax(y_validate[i])
    argmax_pred = np.argmax(y_predicted[i])
    if argmax_val != argmax_pred:
        indexes.append(i)
        values.append(argmax_pred)

print("len1 = ", len(y_predicted))
print("len2 = ", len(y_validate))

#plot_random_with_prediction_v(8, 8, X, y, Yhat, figsize=(8,8))
print("y_validate,shape ",   y_validate.shape )
print("y_prediciton,shape ", y_predicted.shape )


#errors = np.where(y_validate != y_predicted)
errors = (indexes, values)
print(type(errors))
#print("tuple[0]", errors[0])
#print("tuple[1]", errors[1])

print("len tuple[0]", len(errors[0]))
print("len tuple[1]", len(errors[1]))

#print("VALUES =", values)

#sys.exit()

print_statistics(errors, X_validate, y_validate, y_predicted)
