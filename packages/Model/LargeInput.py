from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, ZeroPadding2D
from keras.models import Sequential

def AlexNet_1():
    # Initialize model
    model = Sequential()

    c1 = Conv2D(filters = 96, kernel_size = 11, input_shape = (227, 227, 3), strides = 4, activation = 'relu', name = 'c1')
    mp1 = MaxPool2D(pool_size = 3, strides = 2, name = 'mp1')

    zp2 = ZeroPadding2D((2, 2), name = 'zp1')
    c2 = Conv2D(filters = 256, kernel_size = 5, activation = 'relu', name = 'c2')
    mp2 = MaxPool2D(pool_size = 3, strides = 2, name = 'mp2')

    c3 = Conv2D(filters = 384, kernel_size = 3, activation = 'relu', name = 'c3')

    zp4 = ZeroPadding2D((1, 1), name = 'zp2')
    c4 = Conv2D(filters = 384, kernel_size = 3, activation = 'relu', name = 'c4')

    zp5 = ZeroPadding2D((1, 1), name = 'zp3')
    c5 = Conv2D(filters = 256, kernel_size = 3, activation = 'relu', name = 'c5')
    mp5 = MaxPool2D(pool_size = 3, strides = 2, name = 'mp3')

    # Layer 1
    model.add(c1)
    model.add(mp1)

    # Layer 2
    model.add(zp2)
    model.add(c2)
    model.add(mp2)

    # Layer 3
    model.add(c3)

    # Layer 4
    model.add(zp4)
    model.add(c4)

    # Layer 5
    model.add(zp5)
    model.add(c5)
    model.add(mp5)

    # Layer 6
    model.add(Flatten(name = 'flatten'))
    model.add(Dense(4096, activation = 'relu', name = 'd1'))

    # Layer 7
    model.add(Dense(4096, activation = 'relu', name = 'd2'))

    # Layer 8
    model.add(Dense(10, activation = 'relu', name = 'output'))

    return model