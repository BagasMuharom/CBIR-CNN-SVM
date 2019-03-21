import sys
sys.path.append('../../')

from packages.Utility.Gabor import getGaborFilterBanks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Activation

class CaffeNet:
    
    def __init__(self):
        self.__model = None
        self.__initModel()
        
    def __initModel(self):
        self.__model = Sequential()

        c1 = Conv2D(name='c1', filters = 96, kernel_initializer = getGaborFilterBanks,
                    kernel_size = 11, strides = 4, padding = 'valid', activation = 'relu', 
                    input_shape = (227, 227, 3))
        c2 = Conv2D(name='c2', filters = 256, kernel_initializer = getGaborFilterBanks, 
                    kernel_size = 5, strides = 1, padding = 'same', activation = 'relu')
        c3 = Conv2D(name='c3', filters = 384, kernel_initializer = getGaborFilterBanks, 
                    kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        c4 = Conv2D(name='c4', filters = 384, kernel_initializer = getGaborFilterBanks, 
                    kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
        c5 = Conv2D(name='c5', filters = 256, kernel_initializer = getGaborFilterBanks, 
                    kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')

        self.__model.add(c1)
        self.__model.add(MaxPool2D(pool_size = 2))
        self.__model.add(c2)
        self.__model.add(MaxPool2D(pool_size = 2))
        self.__model.add(c3)
        self.__model.add(c4)
        self.__model.add(c5)
        self.__model.add(MaxPool2D(pool_size = 13))
        self.__model.add(Flatten())
        
    def getModel(self):
        return self.__model

    def predict(self, x):
        return self.__model.predict(x)