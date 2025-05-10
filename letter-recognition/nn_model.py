#from utils import *

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam

import tensorflow as tf

def genrate_model(layers:list, input_size:int ) -> Sequential :
    model = Sequential( [tf.keras.Input(shape=(input_size,)),], name = "my_model")
    for l in layers:
        model.add( Dense(l, activation="sigmoid") )

    model.summary()
    model.compile( loss= BinaryCrossentropy(), optimizer= Adam(0.001),)
    return model



