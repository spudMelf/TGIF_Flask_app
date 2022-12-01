import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
import math

float_features = [str(x) for x in ["Poop poop poop"]]
features = [np.array(float_features)]
tf_model = tf.keras.models.load_model('akin/my_model')
prediction = tf_model.predict(features)
print(prediction)