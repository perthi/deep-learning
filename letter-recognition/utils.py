import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.axes._axes as Axes
from sklearn.model_selection import train_test_split
from keras.api.utils import to_categorical

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
    entries will therefore contain only contain images of zeroes and ones
    """
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    #X = X[0:1000]
    #y = y[0:1000] 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=TEST_SIZE_FRACTION, random_state= None )
    y_train_encoded = to_categorical(y_train)
    y_val_encoded = to_categorical(y_test)
    return X_train, y_train_encoded, X_test, y_val_encoded 
   # return X_train, y_train, X_test, y_test 


def print_statistics(X, y_validate, y_predicted, plot_misclassified: bool = True):
    y_pred_err = []
    y_val_err = []
    indexes_err = []

    y_pred = []
    y_val = []

    m,n = y_predicted.shape

    for i in range(m):
        y_pred.append(np.argmax(y_predicted[i]))
        y_val.append(np.argmax(y_validate[i]))


    for i in  range(m):
        max_pred = np.argmax(y_predicted[i])
        max_val = np.argmax(y_validate[i])
        #print("\n")
        ##print(Yhat[i])    
        if( max_pred != max_val ):
            y_pred_err.append(max_pred)
            y_val_err.append(max_val)
            indexes_err.append(i)
            #print("idx ={} len = {}, argmax_pred = {}, argmax_val = {}".format(i, len(y_predicted[i]), max_pred, max_val) )

    len_err = len(y_val_err)
    len_all = len(y_validate)
    percent_err = 100*(len_err/len_all)
    percent_ok = 100 - percent_err
    print("Success rate: {} %".format(percent_ok))
    print("{} out of {} images was miss classified ( {} %)".format(len_err, len_all, percent_err))

    #sys.exit()

    if plot_misclassified == True:
        def plot_single_image(index):
            print("index = ", index )
            X_random_reshaped = X[index].reshape((npixels_x, npixels_y)).T
            plt.imshow(X_random_reshaped, cmap=None)
            plt.title(f"actual {y_val[index]}, predicted {y_pred[index]}")
            plt.axis('off')

        for i in indexes_err:
            plot_single_image(i)
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