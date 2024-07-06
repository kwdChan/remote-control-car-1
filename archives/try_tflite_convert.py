from data_collection.data_collection import LoggerSet, Logger
from multiprocessing import Manager

import numpy as np
import pandas as pd
import plotly.express as px
from data_collection.video_data import get_frame_iterator
import tensorflow as tf
import tensorflow.keras as keras #type: ignore


def get_model(lr=.1):
    tf.keras.backend.clear_session() #type: ignore
    image_shape = 64, 114, 3
    
    model = keras.Sequential([
        keras.layers.InputLayer(image_shape), 
        keras.layers.Conv2D(16, 3, activation='relu'), 
        keras.layers.MaxPooling2D(),
        
        keras.layers.Conv2D(32, 3, activation='relu'), 
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, 3, activation='relu'), 
        keras.layers.MaxPooling2D(),
        
        keras.layers.Conv2D(64, 3, activation='relu'), 
        keras.layers.MaxPooling2D(),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'), 
        keras.layers.Dense(64, activation='relu'), 
        keras.layers.Dense(4, activation='softmax'), 
    ])

    optimiser = keras.optimizers.Adam(lr)
    model.compile(optimizer=optimiser, loss='Huber', metrics=['MAE'])

    return model 

model = get_model()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model) #type: ignore