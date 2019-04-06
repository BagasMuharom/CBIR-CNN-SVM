import sys
sys.path.append('../../')
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, ZeroPadding2D
from keras.models import Sequential
from packages.Utility.Gabor import GaborFilterBanks, RotatedGaborParams, ChannelizeGaborParams, RandomGaborParams, IterateChannelParams
from packages.Model.BaseModel import BaseModel

class AlexNet1(BaseModel):

    def initModel(self):
        self.initKernelInitializer()
        # Initialize model
        model = Sequential()

        c1 = Conv2D(96, kernel_size = 11, input_shape = (227, 227, 3), strides = 4, activation = 'relu', name = 'c1', kernel_initializer = self.kernels[1])
        mp1 = MaxPool2D(3, strides = 2, name = 'mp1')

        c2 = Conv2D(256, kernel_size = 5, activation = 'relu', name = 'c2', kernel_initializer = self.kernels[2])
        mp2 = MaxPool2D(3, strides = 2, name = 'mp2')

        c3 = Conv2D(384, kernel_size = 3, activation = 'relu', name = 'c3', padding = 'same', kernel_initializer = self.kernels[3])

        c4 = Conv2D(384, kernel_size = 3, activation = 'relu', name = 'c4', padding = 'same', kernel_initializer = self.kernels[4])

        c5 = Conv2D(256, kernel_size = 3, activation = 'relu', name = 'c5', padding = 'same', kernel_initializer = self.kernels[5])
        mp5 = MaxPool2D(3, strides = 2, name = 'mp3')

        # Layer 1
        model.add(c1)
        model.add(mp1)

        # Layer 2
        model.add(c2)
        model.add(mp2)

        # Layer 3
        model.add(c3)

        # Layer 4
        model.add(c4)

        # Layer 5
        model.add(c5)
        model.add(mp5)

        model.add(Flatten(name = 'flatten'))

        # Fully Connected
        model.add(Dense(4096, activation = 'relu', name = 'd1'))
        model.add(Dense(4096, activation = 'relu', name = 'd2'))
        model.add(Dense(10, activation = 'relu', name = 'output'))

        self.model = model
