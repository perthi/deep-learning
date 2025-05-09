# C2_W1 Utilities
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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