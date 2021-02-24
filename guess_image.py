import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL.ImageOps import invert

class_names = 'abcdefghijklmnopqrstuvwxyz'

print('\nLoading model...')
model = keras.models.load_model('trained_models/archive/cnn.h5')

im_file = 'test.png'

def load_image():
    f = im_file
    print('\nLoading \'{}\''.format(f))
    im = Image.open(f).resize((28,28)).convert('L')
    im = ImageEnhance.Sharpness(im).enhance(0)
    im.save('tmp/image0.png')
    return invert(im)
    
image = np.asarray(load_image()).reshape(1,28,28,1) / 255.0

# To predict things...
prediction = model.predict(image)
print('\nNetwork predicts: {} ({}% sure)'.format(class_names[np.argmax(prediction[0])], (100*np.max(prediction[0])).round(2)))

second_opt = np.sort(prediction[0])[-2]
second_opt_place = np.where(prediction[0]==second_opt)[0][0]
print('Second option: {} ({}% sure)'.format(class_names[second_opt_place], 100*np.max(second_opt).round(3)))
