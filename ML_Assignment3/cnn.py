import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import optimizers
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float64')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float64')

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2),padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2, 2),padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(400, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

pred=model.predict(x_test)

y_pred=[]
for p in pred:
    y_pred.append(np.argmax(p))

misclass=np.zeros(10)
misx=[]
misy=[]
misPred=[]
for i in range(len(x_test)):
    if y_test[i]!=y_pred[i]:
        misx.append(x_test[i])
        misy.append(y_test[i])
        misPred.append(y_pred[i])
        misclass[y_test[i]]+=1

# plt.bar(range(10),misclass)
# plt.show()


# plt.figure(figsize=(20,50))
# for i in range(len(misx)):
#     plt.subplot(len(misx)//5+1,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(np.reshape(misx[i],(28,28)), cmap=plt.cm.binary)
#     plt.xlabel(str(misy[i])+' misclassified as '+str(misPred[i]))
# plt.savefig('missclassified.png')
# plt.show()