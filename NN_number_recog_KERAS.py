import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# print(f'tensor flow has been imported', tf.__version__)

# we begin by loading the data from the MNIST dataset,
# provided on the library keras within tensorflow
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Check the image to have a better understanding of the pixels.
#plt.imshow(x_train[0], cmap='grey')
#plt.title(f' y_train{0}')
#plt.show()

# 0 correspods to full black and 255 to full white pixels
# Let's normalize it. We have the data from 0 to 1. (similar to %)
x_train = x_train/255.0
x_test = x_test/255.0

from tensorflow import keras as k
from keras.models import Sequential 
from keras.layers import Dense, Flatten # useful for layers

# Sequential module defines the stack of layers of our NN
# Flatten it transforms our 28x28 pixels into a 784 1D array.
# Imagine the shape it should have. We can add several hidden layers
model = Sequential([Flatten(input_shape = (28,28)),
                    Dense(128, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(10, activation='sigmoid')
                    ])

# 'adam' Adjusts the learning rate while training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# epochs are the number of iterations of the training data
# we can also use part of our data for validation. 
# It is barely a check on our training NN on unseen data

# The batch_size = 32 by default. aka the number of images used in training at the same time
b_s = int(input('Choose your batch size: '))
print(f'MNIST dataset has 60000, so the batches per epoch are {60000 // b_s}')

model.fit(x_train, y_train, epochs=5, batch_size=b_s,validation_split=0.13)

# In case, you want to clear out the terminal
# os.system('clear') 
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {test_acc}")


