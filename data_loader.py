import numpy as np
import pickle
from zipfile import ZipFile
from matplotlib import pyplot as plt

def unzip_and_load(file): # Not used anymore
    with ZipFile(file, 'r') as my_zip:
        with my_zip.open('{}.npy'.format(file.split('/')[-1][:-4])) as f: 
            return np.load(f)

def load_train_data():
    with ZipFile('data/pickled_data.zip', 'r') as my_zip:
        with my_zip.open('pickled_data/training_images.pkl') as f:
            images = pickle.load(f)
        with my_zip.open('pickled_data/training_labels.pkl') as f:
            labels = pickle.load(f)
    return images, labels
        
def load_evaluation_data():
    with ZipFile('data/pickled_data.zip', 'r') as my_zip:
        with my_zip.open('pickled_data/evaluation_images.pkl') as f:
            images = pickle.load(f)
        with my_zip.open('pickled_data/evaluation_labels.pkl') as f:
            labels = pickle.load(f)
    return images, labels

def load_expanded_data():
    with ZipFile('data/expanded_data.zip', 'r') as my_zip:
        with my_zip.open('expanded_data/expanded_training_images.pkl') as f:
            images = pickle.load(f)
        with my_zip.open('expanded_data/expanded_training_labels.pkl') as f:
            labels = pickle.load(f)
    return images, labels

def load_user_data(person=None):
    if person == None:
        with open('data/user_data/train_images.pkl', 'rb') as f:
            images = pickle.load(f)
        with open('data/user_data/train_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    else:
        with open('data/user_data/person{}/train_images.pkl'.format(person), 'rb') as f:
            images = pickle.load(f)
        with open('data/user_data/person{}/train_labels.pkl'.format(person), 'rb') as f:
            labels = pickle.load(f)
            
    return images, labels

def combine_data():
    images1, labels1 = load_user_data(1)
    images2, labels2 = load_user_data(2)
    images = np.vstack((images1,images2))
    labels = np.append(labels1,labels2)

    with open('data/user_data/train_images.pkl', 'wb') as f:
        pickle.dump(images, f)
    with open('data/user_data/train_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
def revert_data(data):
    data = np.multiply(data,255.0)#.reshape(len(data),28,28)
    return data

def revert_label(data):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    return alphabet[int(np.where(data==1)[0])] # Not very fast, but it works. 

if __name__ == '__main__':
    combine_data()
    images, labels = load_user_data()
    print(len(images))
    #images = revert_data(images)
    #print(images.shape)
    #print(labels[20])

    #plt.imshow(images[20], cmap=plt.cm.binary)
    #plt.savefig('test.png')


