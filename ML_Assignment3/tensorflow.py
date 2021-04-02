import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import optimizers
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('2017EE10544.csv',header=None)
data=np.array(data)
n=3000
m=784
x=data[0:n,0:m]
t=data[0:n,784:785]
x=np.array(x)
x=x/1000
t=np.array(t)

def crossAccuracy(folds,nodes):
    b=n//folds
    ac=0
    y=[]
    y1=[]
    f=0
    for i in range(folds):
        temp=np.array(x[f:f+b,0:m])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,m))
        t_train=np.reshape([],(0,1))
        
        for j in range(folds):
            if i!=j:
                x_train=np.concatenate((x_train,y[j]),axis=0)
                t_train=np.concatenate((t_train,y1[j]),axis=0)
                
        
        model = keras.Sequential([
            keras.layers.Dense(nodes, activation='sigmoid'),
#             keras.layers.Dense(200, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax')
        ])
        sgd = optimizers.SGD(lr=0.2, decay=0, momentum=0.9, nesterov=True)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        myModel=model.fit(x_train, t_train,validation_data=(x_test,t_test),batch_size=1,epochs=25)
        test_loss, test_acc = model.evaluate(x_test,  t_test, verbose=2)
        ac+=test_acc
        
        break
        
    return ac

def batch(folds,nodes):
    b=n//folds
    ac=0
    y=[]
    y1=[]
    f=0
    for i in range(folds):
        temp=np.array(x[f:f+b,0:m])
        temp1=np.array(t[f:f+b])
        f=f+b
        y.append(temp)
        y1.append(temp1)
    
    for i in range(folds):
        x_test=y[i]
        t_test=y1[i]
        x_train=np.reshape([],(0,m))
        t_train=np.reshape([],(0,1))
        
        for j in range(folds):
            if i!=j:
                x_train=np.concatenate((x_train,y[j]),axis=0)
                t_train=np.concatenate((t_train,y1[j]),axis=0)
                
        
        model = keras.Sequential([
            keras.layers.Dense(nodes, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        myModel=model.fit(x_train, t_train,validation_data=(x_test,t_test),batch_size=1,epochs=25)
        history=myModel.history['val_accuracy']
        plt.plot(range(1,26),history,label='Batch Size=1')
        
        
        
#         model = keras.Sequential([
#             keras.layers.Dense(nodes, activation='sigmoid'),
#             keras.layers.Dense(10, activation='softmax')
#         ])
#         model.compile(optimizer='adam',
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
#         myModel=model.fit(x_train, t_train,validation_data=(x_test,t_test),batch_size=500,epochs=25)
#         history=myModel.history['val_accuracy']
#         plt.plot(range(1,26),history,label='Batch Size=500') 
        
        
        
#         model = keras.Sequential([
#             keras.layers.Dense(nodes, activation='sigmoid'),
#             keras.layers.Dense(10, activation='softmax')
#         ])
#         model.compile(optimizer='adam',
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
#         myModel=model.fit(x_train, t_train,validation_data=(x_test,t_test),batch_size=1000,epochs=25)
#         history=myModel.history['val_accuracy']
#         plt.plot(range(1,26),history,label='Batch Size=1000') 
        
        
        
        
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Validation Accuracy')
        plt.legend(loc='lower center')
        break
        

batch(4,50)

batch(4,50)