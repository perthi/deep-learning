#from utils import *

from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam
from keras.api.regularizers import *
from keras.api.utils import to_categorical
import sys
from config import *

import tensorflow as tf

def generate_model(layers:list, input_size:int ) -> Sequential :
    model = Sequential( [tf.keras.Input(shape=(input_size,))], name = "my_model")
    
    i = 0
    length = len(layers)
    for l in layers:
        if i != (length -1 ):
           model.add( Dense(l, kernel_regularizer=l2(0.00), activation ="sigmoid") )
        else:
           model.add( Dense(l, kernel_regularizer=l1(0.000),activation ="softmax") )
        i+=1;    
        

    model.summary()
    model.compile( loss= BinaryCrossentropy(), optimizer= Adam(ADAM),)
    return model

