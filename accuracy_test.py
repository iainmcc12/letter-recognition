import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import data_loader
import collections

class_names = 'abcdefghijklmnopqrstuvwxyz'

print('\nLoading model...')

model1 = keras.models.load_model('trained_models/archive3/cnn1.h5')
model2 = keras.models.load_model('trained_models/archive3/cnn2.h5')
model3 = keras.models.load_model('trained_models/archive3/cnn3.h5')
model4 = keras.models.load_model('trained_models/archive3/cnn4.h5')
model5 = keras.models.load_model('trained_models/archive3/cnn5.h5')
model6 = keras.models.load_model('trained_models/archive3/cnn6.h5')
model7 = keras.models.load_model('trained_models/archive3/cnn7.h5')
images, labels = data_loader.load_evaluation_data()
print(images.shape)
errors = []

print('\nEvaluating...')
for i in range(len(images)):
    image = images[i].reshape(1,28,28,1)
    prediction = model1.predict(image)
    prediction += model2.predict(image)
    prediction += model3.predict(image)
    prediction += model4.predict(image)
    prediction += model5.predict(image)
    prediction += model6.predict(image)
    prediction += model7.predict(image)
    prediction_num = np.argmax(prediction, axis=1)
    if prediction_num != labels[i]:
        errors.append(class_names[labels[i]])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.savefig('tmp/image{}({}).png'.format(i,class_names[labels[i]]))

print('\nGot a score of {} out of {} ({}%)'.format(len(images)-len(errors),len(images),round((len(images)-len(errors))/len(images)*100,2)))

frequent_errors = collections.Counter(errors)

plt.figure(figsize=(9, 4))
plt.title('Most Frequent Errors Made by Network')
plt.xlabel('Letters')
plt.ylabel('Errors Made')
plt.bar(range(len(frequent_errors)), list(frequent_errors.values()))
plt.xticks(range(len(frequent_errors)), list(frequent_errors.keys()))
plt.show()
