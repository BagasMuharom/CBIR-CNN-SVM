from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Sequential

def LeNet_1():
    model = Sequential()

    c1 = Conv2D(name='c1', filters = 6,
                kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu', 
                input_shape = (32, 32, 1))
    mp1 = MaxPool2D(pool_size = 2, name = 'mp1')

    c2 = Conv2D(name='c2', filters = 16, 
                kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')
    mp2 = MaxPool2D(pool_size = 2, name = 'mp2')

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

def Cifar_1():
    model = Sequential()

    c1 = Conv2D(32, name = 'c1', kernel_size = 3, padding='same', input_shape = (32, 32, 3))

    c2 = Conv2D(32, name = 'c2', kernel_size = 3, padding = 'same')
    mp2 = MaxPool2D(pool_size = 2)

    c3 = Conv2D(64, name = 'c3', kernel_size = 3, padding = 'same')

    c4 = Conv2D(64, name = 'c4', kernel_size = 3, padding = 'same')
    mp4 = MaxPool2D(pool_size = 2, name = 'mp4')

    c5 = Conv2D(128, name = 'c5', kernel_size = 3, padding = 'same')

    c6 = Conv2D(128, name = 'c6', kernel_size = 3, padding = 'same')
    mp6 = MaxPool2D(pool_size = 2, name = 'mp6')

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

    return model

def Cifar_2():
    pass