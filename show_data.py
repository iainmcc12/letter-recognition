# This program plots 25 images from the training data to show that things are working
print('Loading...')
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
from data import data_loader

train_images = data_loader.revert_data(data_loader.load_train_images())
train_labels = data_loader.load_train_labels()

print('Plotting...')
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(data_loader.revert_label(train_labels[i]))
print('Saving...')
plt.savefig('test.png')
