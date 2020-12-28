print('Loading...')
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from data import data_loader

train_images = data_loader.load_train_images().reshape(124700,28,28) 
train_labels = data_loader.load_train_labels()
test_images = data_loader.load_evaluation_images().reshape(100,28,28)
test_labels = data_loader.load_evaluation_labels()
class_names = 'abcdefghijklmnopqrstuvwxyz'

print('Building model...')
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(26, activation=tf.nn.softmax)
])

#print(model.summary())

print('Compiling model...')
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print('Training...')
model.fit(train_images, train_labels, epochs=1)

# Test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# Save the weights
print('Saving weights...')
model.save_weights('my_checkpoint')

# Testing
#predictions = model.predict(test_images)
#print(class_names[np.argmax(predictions[22])]) # <-- model's prediction
#print(data_loader.revert_label(test_labels[22])) # <-- what it actually is


