import flask
from flask import Flask, request, jsonify

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory

import os
import matplotlib.image as mpimg

app = Flask(__name__)

# cat_dog_model = tf.keras.models.load_model("catvsdog.h5")

def predict_from_model():
    #......
    return True

@app.route('/', methods = ["GET"])
def basic():  
    prediction = predict_from_model()
    return jsonify({'prediction': prediction})


if __name__ == 'main':
    app.run(debg=True)