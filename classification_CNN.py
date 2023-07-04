from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


import numpy as np
def data_preprocess():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    print(train_images)
    train_images = train_images.astype('float32') / 255
    #print(train_images[0])
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images,train_labels,test_images,test_labels

def data_preprocessing():
    data = pd.read_csv("DATA_TEST.csv", sep=',', encoding='utf-8')
    #print(data.shape[0])
    data_train = data.iloc[:, 1:]
    min_max_scaler = preprocessing.MinMaxScaler()
    data_train = min_max_scaler.fit_transform(data_train)
    #print(data_train)
    data_label = data.iloc[:, 0]
    print(data_label.shape)
    x_train, x_test, y_train, y_test = train_test_split(data_train, data_label, test_size=0.2)
    #print(x_train)
    return x_train, x_test, y_train, y_test



#搭建网络
def build_module():
    model = models.Sequential()
    #第一层卷积层
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(95,95,1)))
    #第二层最大池化层
    model.add(layers.MaxPooling2D((2,2)))
    #第三层卷积层
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    #第四层最大池化层
    model.add(layers.MaxPooling2D((2,2)))
    #第五层卷积层
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    #第六层Flatten层，将3D张量平铺为向量
    model.add(layers.Flatten())
    #第七层全连接层
    model.add(layers.Dense(64, activation='relu'))
    #第八层softmax层，进行分类
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
    train_images,train_labels,test_images,test_labels=data_preprocess()
    model=build_module()
    print(model.summary())
    model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(train_images, train_labels, epochs = 5, batch_size=64)
    draw_loss(history)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_loss=',test_loss,'  test_acc = ', test_acc)


