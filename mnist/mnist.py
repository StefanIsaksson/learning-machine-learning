# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Training and testing data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Explore the data
print(train_images.shape)
print(len(train_images))
print(train_labels.shape)
print(len(train_labels))


## Show a picture
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Print 10x10 pixels of top right corner of first picture
first_picture = train_images[0];
print(type(first_picture))
print(first_picture[0:10, 0:10])

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

## Display the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Print 10x10 pixels of top right corner of first picture
print(first_picture[0:10, 0:10])

# Build the model
## Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (choose 5 epochs)
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)
print(test_labels[0])
print(predictions[0])
print(np.argmax(predictions[0]))

## Display the first 25 test images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Act: " + str(test_labels[i]) + ",  Pre: " + str(np.argmax(predictions[i])))
plt.show()

## Show prediction of first test_label in grapth
plt.bar(range(10), predictions[0]*100, align='center', alpha=0.5)
plt.ylabel('Probability (%)')
plt.title('Predication of first picture')
plt.show()