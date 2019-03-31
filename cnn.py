import numpy as np 
import matplotlib.pyplot as plt 

import keras 
from keras.layers import Conv2D,Dense,Input,Activation,Dropout,BatchNormalization,Flatten,MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import datasets

def cnn():
    inputs = Input(shape=(28,28,1))
    xx = Conv2D(32,(3,3))(inputs)
    xx = Activation("relu")(xx)

    xx = Conv2D(32,(3,3))(xx)
    xx = Activation("relu")(xx)

    xx = MaxPooling2D(pool_size=(2,2))(xx)
    xx = Dropout(0.25)(xx)

    xx = Conv2D(64,(3,3))(xx)
    xx = Activation("relu")(xx)
    xx = Conv2D(64,(3,3))(xx)
    xx = Activation("relu")(xx)

    xx = MaxPooling2D(pool_size=(2,2))(xx)
    xx = Flatten()(xx)

    xx = Dense(512)(xx)
    xx = Activation("relu")(xx)
    xx = Dropout(0.5)(xx)

    xx = Dense(2)(xx)
    outputs = Activation("softmax")(xx)

    model = Model(input=inputs,output=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    return model

if __name__=="__main__":
    (x_train,y_train),(x_test,y_test) = \
        datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # one-hot vector形式に変換する
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    mymodel = cnn()
    mymodel.fit(x_train, y_train,batch_size=128,epochs=3,verbose=1,)

    score = mymodel.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
