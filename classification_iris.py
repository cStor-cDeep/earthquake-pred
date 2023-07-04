from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

import numpy as np

def data_preprocessing():

    data = pd.read_csv("iris_training.csv", sep=',', encoding='utf-8')
    print(data)
    data_train = data.iloc[:, :-1]
    print(data_train)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_train = min_max_scaler.fit_transform(data_train)
    #print(data_train)
    data_label = data.iloc[:, -1]
    print(data_label.shape)
    x_train, x_test, y_train, y_test = train_test_split(data_train, data_label, test_size=0.2)
    #print(x_train)
    return x_train, x_test, y_train, y_test

def ANN_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu',input_shape = (4,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(7, activation='softmax'))
    return model


def draw_loss(history):
    loss=history.history['loss']
    epochs=range(1,len(loss)+1)
    plt.subplot(1,2,1)#第一张图
    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.title("Training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)#第二张图
    accuracy=history.history['accuracy']
    plt.plot(epochs,accuracy,'bo',label='Training accuracy')
    plt.title("Training accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.suptitle("Train data")
    plt.legend()
    plt.show()
if __name__=='__main__':
    x_train, x_test, y_train, y_test = data_preprocessing()
    model =ANN_model()
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss =  tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=200, batch_size=64)
    draw_loss(history)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('test_loss=',test_loss,'  test_acc = ', test_acc)


