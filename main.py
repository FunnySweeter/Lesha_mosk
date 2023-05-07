from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import numpy

import tensorflow
from tensorflow import random

# model = load_model("mnist.h5")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("Размерность x_train:", x_train.shape[0])

batch_size = 100
epochs = 100

model = Sequential(
   [
       Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
       Conv2D(64, (3, 3), activation="relu"),
       MaxPooling2D(pool_size=(2, 2)),

       Dropout(0.25),
       Flatten(),

       Dense(512, "relu"),
       Dropout(0.25),
       Dense(256, "relu"),
       Dropout(0.25),
       Dense(num_classes, "softmax")
   ]
)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])

model.summary()

hist = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
model.save("mnist.h5")

score = model.evaluate(x_test, y_test, verbose = 0)
print('Потери на тесте: ', score[0], '\nТочность на тесте: ', score[1])

# model = Sequential(
#     [
#         Dense(1, input_shape=(1,), activation='relu')
#     ]
# )

# model.summary()
