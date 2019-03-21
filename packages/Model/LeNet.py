import sys
sys.path.append('../../')

from packages.Utility.Gabor import getGaborFilterBanks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Activation, GlobalAveragePooling2D

class LeNet:
    
    def __init__(self, depth = 1):
        self.__model = None
        self.__depth = depth
        self.__initModel()
        
    def __initModel(self):
        self.__model = Sequential()

        c1 = Conv2D(name='c1', filters = 6, kernel_initializer = getGaborFilterBanks,
                    kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu', 
                    input_shape = (32, 32, self.__depth))
        c2 = Conv2D(name='c2', filters = 16, kernel_initializer = getGaborFilterBanks, 
                    kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')

        self.__model.add(c1)
        self.__model.add(MaxPool2D(pool_size = 2))
        self.__model.add(c2)
        self.__model.add(MaxPool2D(pool_size = 2))
        self.__model.add(GlobalAveragePooling2D())
        self.__model.add(Flatten())
        
    def getModel(self):
        return self.__model

    def predict(self, x):
        return self.__model.predict(x)