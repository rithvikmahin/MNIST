import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.utils import to_categorical
import numpy as np


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train/256
X_test = X_test/256
print(np.shape(X_train))

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Neural Network
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=784))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=8, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print("Loss - ", score[0])
print("Accuracy - ", score[1])

model.save("MNIST_model.h5")