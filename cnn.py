# A convolutional neural network based on LeNet-5 architecture
print('Loading...')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pickle

train_images, train_labels = data_loader.load_train_data()
test_images, test_labels = data_loader.load_evaluation_data()
train_images = train_images.reshape(len(train_images),28,28,1)
test_images = test_images.reshape(len(test_images),28,28,1)

class_names = 'abcdefghijklmnopqrstuvwxyz'
checkpoint_path = 'trained_models/cnn/cp.ckpt'


print('Building model...')
model = keras.Sequential()

model.add(layers.Conv2D(32,kernel_size = 3, activation='relu', input_shape=[28,28,1]))

model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,kernel_size = 3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(64,kernel_size = 3, activation='relu')) 
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,kernel_size = 3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))          

model.add(layers.Conv2D(128,kernel_size = 4, activation='relu'))          
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(26, activation='softmax'))


if input('\nLoad weights from \'{}\' and continue training? (y/n) '.format(checkpoint_path)) == 'y':
    model.load_weights(checkpoint_path) # loads checkpoint 

print('\nCompiling model...')
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

# Create callback to save the model after each epoch
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Decrease learning rate each epoch
annealer = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# Generate more training images
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 5,
    zoom_range = 0.10,
    width_shift_range=0.1, 
    height_shift_range=0.1
)
datagen.fit(train_images)

# Main
while True:
    num_epochs = int(input('\nTrain for how many epochs? '))
    print('\nTraining...')
    model.fit(datagen.flow(train_images, train_labels, batch_size=64), epochs=num_epochs,
                        steps_per_epoch = train_images.shape[0]//64, 
                        callbacks = [cp_callback, annealer])
    
    # Test accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy:', test_acc)

    if input('\nTrain for more epochs? (y/n) ') != 'y':
        if input('Save model? (y/n) ') == 'y':
            print('Saving entire model...')
            model.save('trained_models/cnn/cnn.h5')
            break
