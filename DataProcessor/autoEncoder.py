#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:18:00 2019

@author: mhit
"""

from keras import optimizers
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import math
import argparse

def autoencoder(epochs, data, output):
    
    data = np.load(data)
    n_in = data.shape[1]
    
    
    bottleNeck = 3
    step = math.ceil((n_in - bottleNeck) / 4)
    h1 = n_in - step
    h2 = h1 - step
    h3 = h2 - step
    
    # define model
    inputs = Input(shape=(n_in,))
    encoded = Dense(h1,activation="relu")(inputs)
    encoded = Dense(h2,activation="relu")(encoded)
    encoded = Dense(h3,activation="relu")(encoded)

    encoded = Dense(bottleNeck,activation="relu")(encoded)

    decoded = Dense(h3,activation="relu")(encoded)
    decoded = Dense(h2,activation="relu")(decoded)
    decoded = Dense(h1,activation="relu")(decoded)    
    decoded = Dense(n_in, activation="sigmoid")(decoded)
    
    
    
    model = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [
        ModelCheckpoint('model.h5', monitor='loss', save_best_only=True, verbose=0, save_weights_only=False, mode='auto', period=1)
    ]
    model.fit(data, data, epochs=epochs, shuffle=False, batch_size=32, callbacks=callbacks)
    model = load_model('model.h5')
    encoder = Model(model.input, model.layers[-5].output)
    # demonstrate recreation
    result = encoder.predict(data)
    np.save(output, result)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-e', '--epochs', action="store", default=200, type=int)
   parser.add_argument('-i', '--input', action="store")
   parser.add_argument('-o', '--output', action="store")
   args = parser.parse_args()
   autoencoder(args.epochs, args.input, args.output)






