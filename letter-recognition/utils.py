import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.axes._axes as Axes
from sklearn.model_selection import train_test_split
from config import TEST_SIZE_FRACTION
import sys

npixels_x:int = 20
npixels_y:int = 20
input_size:int = npixels_x*npixels_y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data():
    """!
    Loading data, there are 500 samples of each digit
    which is organized form 0 to 9. The first 1000
    entries will therfore contain only contain images of zeroes and ones
    """
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000] 
    X_train, X_validate, y_train, y_validate = train_test_split( X, y, test_size=TEST_SIZE_FRACTION, random_state= None )
    
    return X_train, y_train, X_validate, y_validate 


def print_statistics(errors, X, y, Yhat, plot_misclassified: bool = True):
    print("tuple[0]", errors[0])
    print("tuple[1]", errors[1])
    indexes = errors[0]
    values =  errors[1]

    for i in range(len(indexes)):
        print("index = {}, value = {}, yhat = {}".format(indexes[i], values[i], Yhat[ indexes[i] ] ))

    len_err = len(values)
    len_all = len(y)
    percent_err = 100*(len_err/len_all)
    percent_ok = 100 - percent_err
    print("Success rate: {} %".format(percent_ok))
    print("{} out of {} images was miss classified ( {} %)".format(len_err, len_all, percent_err))

    if plot_misclassified == True:
        def plot_single_image(index):
            print("index = ", index )
            X_random_reshaped = X[index].reshape((npixels_x, npixels_y)).T
            plt.imshow(X_random_reshaped, cmap='gray')
            plt.title(f"actual {y[index,0]}, predicted {Yhat[index, 0]}")
            plt.axis('off')

        for i in range(len(values)):
            plot_single_image(indexes[i])
            plt.show()


# Pick random indexes from and npy array and plot them in a grid
def plot_random(rows:int, columns:int, X, y, figsize=(8,8)):
    fig, axes = plt.subplots(rows,columns, figsize=figsize)
    fig.tight_layout(pad=0.1)
    m, n = X.shape

    for i,ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        X_random_reshaped = X[random_index].reshape((npixels_x,npixels_y)).T    # reshape the image
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
        X_random_reshaped = X[random_index].reshape((npixels_x,npixels_y)).T
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
    
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,input_size))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0
    # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}")
        ax.set_axis_off()

    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()   
    
    return fig


def plot_random_with_prediction_v(rows:int, columns:int, X, y, Yhat, figsize=(8,8)):
    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        X_random_reshaped = X[random_index].reshape((npixels_x, npixels_y)).T  # reshape the image
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray') # Display the image
   
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
        ax.set_axis_off() 
    fig.suptitle("Label, Yhat", fontsize=16)
    plt.show()
    return fig


def print_params(L1,L2,L3, L4):
    L1_num_params = input_size *L1 + L2  # W1 parameters  + b1 parameters
    L2_num_params = L1*L2 + L2   # W2 parameters  + b2 parameters
    L3_num_params = L2 *L3 + L3
    L4_num_params = L3 * 1 + 1     # W3 parameters  + b3 parameters

    print("L1 params = ", L1_num_params, ", L2 params = ", 
      L2_num_params, ",  L3 params = ", L3_num_params, ",  L4 params = ", L4_num_params)