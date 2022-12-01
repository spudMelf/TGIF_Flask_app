import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import math

# Create flask app
flask_app = Flask(__name__, static_folder='staticFiles')

tf_model = tf.keras.models.load_model('./NLP_Model')

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [str(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = tf_model.predict(features)
    
    

    # custom function
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # define vectorized sigmoid
    sigmoid_v = np.vectorize(sigmoid)

    y_new = sigmoid_v(np.array([prediction[0]]))

    y_new = y_new * 100

    y_new_text = '{:.0f}%'.format(int(y_new[0]))

    return render_template("index.html", prediction_text = "The model computes that the bill will pass with likelihood:  {}".format(y_new_text))

if __name__ == "__main__":
    flask_app.run(debug=True, port=4000)