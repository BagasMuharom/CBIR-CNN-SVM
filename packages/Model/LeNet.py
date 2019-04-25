import sys
sys.path.append('../../')
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten, ZeroPadding2D
from keras.models import Sequential
from packages.Model.BaseModel import BaseModel
from packages.Utility.Gabor import GaborFilterBanks, RotatedGaborParams, ChannelizeGaborParams, RandomGaborParams

class LeNet(BaseModel):

    def initModel(self):
        model = Sequential()

        c1 = Conv2D(name='c1', filters = 6, kernel_size = 5, activation = 'relu', 
                    input_shape = (32, 32, 1), kernel_initializer = self.kernels[1])
        mp1 = MaxPooling2D(pool_size = 2, name = 'mp1')

        c2 = Conv2D(name='c2', filters = 16, kernel_size = 5, activation = 'relu', kernel_initializer = self.kernels[2])
        mp2 = MaxPooling2D(pool_size = 2, name = 'mp2')

        # Layer 1
        model.add(c1)
        model.add(mp1)

        # Layer 2
        model.add(c2)
        model.add(mp2)

        model.add(Flatten(name = 'flatten'))

        # Fully Connected
        model.add(Dense(120, activation = 'sigmoid', name = 'd1'))
        model.add(Dense(84, activation = 'sigmoid', name = 'd2'))
        model.add(Dense(10, activation = 'softmax', name = 'output'))

        self.model = model

class LeNet1_1(BaseModel):

    def initModel(self):
        model = Sequential()

        c1 = Conv2D(name='c1', filters = 6, kernel_size = 5, activation = 'relu', 
                    input_shape = (32, 32, 1), kernel_initializer = self.kernels[1])
        mp1 = MaxPooling2D(pool_size = 2, name = 'mp1')

        c2 = Conv2D(name='c2', filters = 16, kernel_size = 5, activation = 'relu', kernel_initializer = self.kernels[2])
        mp2 = MaxPooling2D(pool_size = 2, name = 'mp2')

        # Layer 1
        model.add(c1)
        model.add(BatchNormalization())
        model.add(mp1)

        # Layer 2
        model.add(c2)
        model.add(BatchNormalization())
        model.add(mp2)

        model.add(Flatten(name = 'flatten'))

        # Fully Connected
        model.add(Dense(120, activation = 'sigmoid', name = 'd1'))
        model.add(Dropout(0.25))
        model.add(Dense(84, activation = 'sigmoid', name = 'd2'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation = 'softmax', name = 'output'))

        self.model = model
        