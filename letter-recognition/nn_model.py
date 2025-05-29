#from utils import *

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.losses import BinaryCrossentropy
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical
import sys

import tensorflow as tf

def generate_model(layers:list, input_size:int ) -> Sequential :
    model = Sequential( [tf.keras.Input(shape=(input_size,)),], name = "my_model")
    i = 0
    length = len(layers)
    for l in layers:
        if i != (length -1 ):
           model.add( Dense(l, activation ="sigmoid") )
           print( "i == length" )
           print("i = {}, len = {}".format(i, length) )
        else:
           model.add( Dense(10, activation ="softmax") )
           print( "i != length" )
           print("i = {}, len = {}".format(i, length) )
        i+=1;    
        
        #print("i = {}, len = {}".format(i, length) )
   
   # print("{} out of {} images was miss classified ( {} %)".format(len_err, len_all, percent_err))

    model.summary()
    model.compile( loss= BinaryCrossentropy(), optimizer= Adam(0.001),)
    #sys.exit()
    return model



