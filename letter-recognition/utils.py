# C2_W1 Utilities
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.axes._axes as Axes
#from sklearn.datasets i
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data(debug: bool = False):
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    if debug == True:
        print ('The first element of X is: ', X[0])
        print ('The shape of X is: ' + str(X.shape))
        print ('The shape of y is: ' + str(y.shape))

    return X, y

def load_weights():
    w1 = np.load("data/w1.npy")
    b1 = np.load("data/b1.npy")
    w2 = np.load("data/w2.npy")
    b2 = np.load("data/b2.npy")
    return w1, b1, w2, b2


def prediction_debug(X, model):
    prediction = model.predict(X[0].reshape(1,400))  # a zero
    print(f" predicting a zero: {prediction}")
    prediction = model.predict(X[500].reshape(1,400))  # a one
    print(f" predicting a one:  {prediction}")

    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print(f"prediction after threshold: {yhat}")

