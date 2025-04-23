import tensorflow as tf
import numpy as np
import itertools

# We load the data from the keras dataset, 
# although we won't use keras this time

from tensorflow.keras.datasets import mnist

# We load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# We normalize it (0-1) instead of (0-255),
# so the range of data is similar.
x_train = x_train / 255.0
x_test = x_test / 255.0

# We convert the data to float32 for TensorFlow.
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)


# Now we reshape the 28 by 28 to a 1D array
# Careful with the reshape method. Check documentation.
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Correctly handle y_train and y_test one-hot encoding for tensors.
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Heere comes the Hyperparameters
# Learning rate is the step size over each iteration.
# Never too big(we will overshoot the minimum), 
# but never too small(it will take ages).
learning_rate = 0.1

# The number of samples in each batch, that will be pass to the training model.
batch_size = 32
# The number of times the training data is seen by the model.
num_epochs = 4


# After some hyperparameter tuning, we found the following values
# 3 hidden layers
# lr = 0.1
# batch_size = 32
# num_epochs = 4
# Test Accuracy of 0.97 on the test set. 
# Training accuracy
# Loss function: 
#Training Accuracy: 0.98089998960495
#Test Accuracy: 0.9714999794960022


# Number of neurons in our layers
num_neurons_input = 28 * 28
num_neurons_hidden_1 = 128
num_neurons_hidden_2 = 64
num_neurons_hidden_3 = 32
num_neurons_output = 10

# Now, we initialize the weights and biases for every connection (n-1), being n the number of layers
W1 = tf.Variable(tf.random.truncated_normal([num_neurons_input, num_neurons_hidden_1], stddev=0.1))
b1 = tf.Variable(tf.zeros([num_neurons_hidden_1]))
W2 = tf.Variable(tf.random.truncated_normal([num_neurons_hidden_1, num_neurons_hidden_2], stddev=0.1))
b2 = tf.Variable(tf.zeros([num_neurons_hidden_2]))
W3 = tf.Variable(tf.random.truncated_normal([num_neurons_hidden_2, num_neurons_hidden_3], stddev=0.1))
b3 = tf.Variable(tf.zeros([num_neurons_hidden_3]))
W4 = tf.Variable(tf.random.truncated_normal([num_neurons_hidden_3, num_neurons_output], stddev=0.1))
b4 = tf.Variable(tf.zeros([num_neurons_output]))

# Feed forward function
def feed_forward(x):
    # clearly define the number hidden layers and the output.
    # ONly on these mentioned the feed forward function will work
    # Layer 1. We use relu activation function
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))  # Correct activation function
    # Layer 2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, W2), b2))
    # Layer 3
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, W3), b3))
    # Output layer
    out_layer = tf.add(tf.matmul(layer_3, W4), b4)

    return out_layer
    # We use the softmax function to get the probabilities

def loss_function(y_true, y_pred):
    # we compute the loss function between true labels and predicited labels.
    # As parameters we pass 2 tensors, y_true and y_pred.
    # We measure the performance of the model using the cross entropy loss function
    # reduce_mean is used to get the mean of the loss function
    # between all the batches. So, we finally get 1 value.
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=y_pred, labels=y_true))

# We need to optimize it.
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

# Let's trin it. Recall the 

# Training function.
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = feed_forward(x_batch)
        loss = loss_function(y_batch, predictions)

    # Compute gradients.
    gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3, W4, b4])
    
    # Update weights and biases.
    optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3, W4, b4]))

    return loss

# Test the model.
def evaluate(x_test, y_test):
    predictions = feed_forward(x_test)  # Correct function name
    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

# Training loop.
for epoch in range(num_epochs):
    num_batches = x_train.shape[0] // batch_size
    for i in range(num_batches):
        batch_x = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]
        loss = train_step(batch_x, batch_y)

    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")



# print accuracy on training set
accuracy = evaluate(x_train, y_train)
print(f"Training Accuracy: {accuracy.numpy()}")


# Evaluate the model on the test set.
accuracy = evaluate(x_test, y_test) 
print(f"Test Accuracy: {accuracy.numpy()}")
