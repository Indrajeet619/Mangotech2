

import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
import glob
import seaborn as sns
import cv2
import io

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import ipywidgets as widgets
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import MaxPooling2D, Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout

# architecture Pare
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from sklearn.metrics import classification_report, confusion_matrix


X  =[] 
Y = [] 
image_size = (227, 227)

for i in labels:
    print(Datasetspath)
    fileRead = glob.glob(Datasetspath + "*")
    print(len(fileRead))
    
    for file in fileRead:
        image = cv2.imread(file)
        img = cv2.resize(image, image_size)
        X.append(img)
        Y.append(i)
X = np.array(X)
Y = np.array(Y)

for i in labels:
    print(Datasetspath)
    fileRead = glob.glob(Datasetspath + "*")
    print(fileRead[1])
X.shape, Y.shape
plt.imshow(X[0])

Temp_y = []
for i in Y:
    Temp_y.append(labels.index(i))
Y = to_categorical(Temp_y)
Y[0]


array([1., 0., 0., 0.], dtype=float32)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=101)

models = Sequential()


#1st Conv2D Layer
models.add(Conv2D(96, kernel_size = (11, 11), strides = (4, 4), 
                 padding = "valid", activation  = 'relu', input_shape = (227, 227, 3)))
models.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2), padding = "valid",
                       data_format = None))



#2nd Conv2D Layer

models.add(Conv2D(256, kernel_size = (5, 5), strides = 1, 
                 padding = "same", activation  = 'relu'))

models.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2), padding = "valid",#"same"
                       data_format = None))




#3rd Conv2D Layer
models.add(Conv2D(384, kernel_size = (3, 3), strides = 1, 
                 padding = "same", activation  = 'relu'))



#4th Conv2D Layer
models.add(Conv2D(384, kernel_size = (3, 3), strides = 1, 
                 padding = "same", activation  = 'relu'))


#5th Conv2D Layer

models.add(Conv2D(256, kernel_size = (3, 3), strides = 1, 
                 padding = "same", activation  = 'relu'))

models.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2), padding = "valid",#"same"
                       data_format = None))


# Flatten Layer
models.add(Flatten())

models.add(Dense(4096, activation = 'relu'))
models.add(Dense(4096, activation = 'relu'))
#models.add(Dense(1000, activation = 'relu'))
models.add(Dense(4, activation = 'softmax'))

models.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)



models.compile(loss = "categorical_crossentropy",
             optimizer = optimizer,
             metrics = ["accuracy"])

history = models.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 10, batch_size = 2, verbose = 1)



ax = plt.gca()
ax.set_ylim([0, 1])
print("Model Accuracy\n")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()






