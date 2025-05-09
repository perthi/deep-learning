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


# Pick random indexes from and npy array and plot them in a grid
def plot_random(rows:int, columns:int, X, y, figsize=(8,8)):
    fig, axes = plt.subplots(rows,columns, figsize=figsize)
    fig.tight_layout(pad=0.1)
    m, n = X.shape

    for i,ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        X_random_reshaped = X[random_index].reshape((20,20)).T    # reshape the image
        ax.imshow(X_random_reshaped, cmap='gray')
        # Display the label above the image
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
    return fig   

def plot_random_with_prediction(rows:int, columns:int, X, y, model, figsize=(8,8)):
    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
    m, n = X.shape

    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        X_random_reshaped = X[random_index].reshape((20,20)).T
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
    
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0
    # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}")
        ax.set_axis_off()
    
    return fig


def plot_random_with_prediction_v(rows:int, columns:int, X, y, Yhat, figsize=(8,8)):
    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        X_random_reshaped = X[random_index].reshape((20, 20)).T  # reshape the image
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray') # Display the image
   
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
        ax.set_axis_off() 
    
    return fig

