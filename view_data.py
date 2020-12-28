# This program allows you to view the data from the emnist cache in order or randomly

print('Loading modules...')
from emnist import extract_training_samples
from matplotlib import pyplot as plt
import random

alphabet = 'abcdefghijklmnopqrstuvwxyz'

print('Loading images and labels...\n')
images, labels = extract_training_samples('letters')
num_images = len(images)

def in_order(start=0):
    for image in range(start,num_images):
        letter = labels[image]
        print('Image {}: {} --> {}'.format(image,letter,alphabet[letter-1]))
        plt.imshow(images[image], cmap='gray')
        plt.show()

def view_random():
    while True:
        image = random.randint(0,num_images)
        letter = labels[image]
        print('Image {}: {} --> {}'.format(image,letter,alphabet[letter-1]))
        plt.imshow(images[image], cmap='gray')
        plt.show()

view_random()
