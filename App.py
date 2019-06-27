from keras.models import load_model
import tensorflow as tf
from flask import request
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template
from keras import backend

app = Flask(__name__)

model = load_model("MNIST_model.h5")
mnist = tf.keras.datasets.mnist
graph = tf.get_default_graph()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train/256
X_test = X_test/256

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        value = request.form['index']
        image = X_test[int(value)]
        image = np.expand_dims(image, axis=0)
        with graph.as_default():
            prediction = model.predict(image)
            return(str(prediction.argmax()))

show_image = np.reshape(X_test[1224], (28, 28))
plt.imshow(show_image, cmap='Greys')
plt.show()

if __name__ == '__main__':
    app.run(debug=True)

