import sys
sys.path.append('../../')
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten, ZeroPadding2D
from keras.models import Sequential
from packages.Model.BaseModel import BaseModel
from packages.Utility.Gabor import GaborFilterBanks, RotatedGaborParams, ChannelizeGaborParams, RandomGaborParams

class CFNet1(BaseModel):

    def initModel(self):
        self.initKernelInitializer()
        model = Sequential()
        
        c1 = Conv2D(32, name = 'c1', kernel_size = 3, padding = 'same', input_shape = (32, 32, 3), kernel_initializer = self.kernels[1])
        
        c2 = Conv2D(32, name = 'c2', kernel_size = 3, padding = 'same', kernel_initializer = self.kernels[2])
        mp2 = MaxPooling2D(pool_size = 2)
        
        c3 = Conv2D(64, name = 'c3', kernel_size = 3, padding = 'same', kernel_initializer = self.kernels[3])
        
        c4 = Conv2D(64, name = 'c4', kernel_size = 3, padding = 'same', kernel_initializer = self.kernels[4])
        mp4 = MaxPooling2D(pool_size = 2, name = 'mp4')
        
        c5 = Conv2D(128, name = 'c5', kernel_size = 3, padding = 'same', kernel_initializer = self.kernels[5])
        
        c6 = Conv2D(128, name = 'c6', kernel_size = 3, padding = 'same', kernel_initializer = self.kernels[6])
        mp6 = MaxPooling2D(pool_size = 2, name = 'mp6')

        # Layer 1
        model.add(c1)

        # Layer 2
        model.add(c2)
        model.add(mp2)
        
        # Layer 3
        model.add(c3)

        # Layer 4
        model.add(c4)
        model.add(mp4)
        
        # Layer 5
        model.add(c5)

        # Layer 6
        model.add(c6)
        model.add(mp6)
        
        model.add(Flatten(name = 'flatten'))
                
        # Fully Connected
        model.add(Dense(10, name = 'output', activation = 'softmax'))

        self.model = model
        