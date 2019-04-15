import numpy as np
import keras.backend as K
from keras.initializers import Initializer

# Membuat filter gabor
def gaborFilter(size, lambd, theta, psi, sigma, gamma):
    gabor = np.zeros(shape = size)
    half_height = int(size[1] / 2)
    half_width = int(size[0] / 2)
    
    for y in range(-half_height, half_height if size[1] % 2 == 0 else half_height + 1):
        for x in range(-half_width, half_width if size[0] % 2 == 0 else half_width + 1):
            x_aksen = x * np.cos(theta) + y * np.sin(theta)
            y_aksen = -x * np.sin(theta) + y * np.cos(theta)
            gabor[y + half_height][x + half_width] = np.exp(-(x_aksen ** 2 + ((gamma ** 2) * (y_aksen ** 2))) / (2 * (sigma ** 2))) * np.cos((((2 * np.pi * x_aksen) / lambd) + psi))

    return gabor

class GaborParams():

    def getParams(self, shape, i, j):
        raise NotImplementedError

class RotatedGaborParams(GaborParams):

    def __init__(self, lambd, sigma, psi, gamma):
        self.lambd = lambd
        self.sigma = sigma
        self.psi = psi,
        self.gamma = gamma

    def getParams(self, shape, i, j):
        multiplier = 360 / shape[3]
        degree = i * multiplier
        theta = degree * np.pi / 180
        
        return self.lambd, theta, self.psi, self.sigma, self.gamma

class IterateChannelParams(GaborParams):
    
    def __init__(self, lambd, sigma, gamma):
        self.lambd = lambd
        self.sigma = sigma
        self.gamma = gammba

    def getParams(self, shape, i, j):
        multiplier = 360 / shape[3]
        degree = multiplier * i
        num_channel = shape[2]
        
        i += 1
        j += 1
        ratio = j / num_channel
        
        lambd =  np.linspace(self.lambd[0], self.lambd[1], num_channel)[j - 1]
        theta = degree * np.pi / 180
        psi = j
        sigma = np.linspace(self.sigma[0], self.sigma[1], num_channel)[j - 1]
        gamma = np.linspace(self.gamma[0], self.gamma[1], num_channel)[j - 1]
        
        return lambd, theta, psi, sigma, gamma

class ChannelizeGaborParams(GaborParams):
    
    def getParams(self, shape, i, j):
        multiplier = 360 / shape[3]
        degree = multiplier * i
        
        i += 1
        j += 1
        
        lambd = (i * j) / shape[2]
        theta = degree * np.pi / 180
        psi = (i + j) / shape[2]
        sigma = (i + j)
        gamma = (i + j) / shape[2]
        
        return lambd, theta, psi, sigma, gamma

class RandomGaborParams(GaborParams):

    def getParams(self, shape, i, j):
        multiplier = 360 / shape[3]
        degree = multiplier * i
        num_channel = shape[2]
        
        i += 1
        j += 1
        
        lambd = np.random.rand() * num_channel
        theta = degree * np.pi / 180
        psi = np.random.rand() * num_channel
        sigma = np.random.rand() * num_channel
        gamma =np.random.rand() * num_channel
        
        return lambd, theta, psi, sigma, gamma

class ComplexGaborParams():
    
    def __init__(self, sigma = -1, lambd = -1, gamma = -1, psi = -1, theta = (-1, -1)):
        self.theta = theta
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
    
    def getParams(self, shape, i, j):
        self.i = i + 1
        self.j = j + 1
        self.shape = shape
        
        theta = self.hitungTheta() * np.pi / 180
        sigma = self.hitungSigma()
        lambd = self.hitungLambda()
        gamma = self.hitungGamma()
        psi = self.hitungPsi()
        
        return lambd, theta, psi, sigma, gamma
    
    def hitungTheta(self):
        start = (360 / self.shape[2]) if self.theta[0] is -1 else self.theta[0]
        end = (360 - (360 / self.shape[2])) if self.theta[1] is -1 else self.theta[1]
        
        rangeTheta = np.linspace(start, end, self.shape[3])
        
        return rangeTheta[self.i - 1]
    
    def hitungSigma(self):
        start = (self.i + 1) if self.sigma is -1 else self.sigma[0]
        end = (self.i + self.j) if self.sigma is -1 else self.sigma[1]
        
        rangeSigma = np.linspace(start, end, self.shape[2])
        
        return rangeSigma[self.j - 1]
    
    def hitungLambda(self):
        start = ((self.i + 1) / self.shape[2]) if self.lambd is -1 else self.lambd[0]
        end = (self.i * self.j / self.shape[2]) if self.lambd is -1 else self.lambd[1]
        
        rangeLambd = np.linspace(start, end, self.shape[2])
        
        return rangeLambd[self.j - 1]
    
    def hitungGamma(self):
        start = ((self.i + 1) / self.shape[2]) if self.gamma is -1 else self.gamma[0]
        end = ((self.i + self.j) / self.shape[2]) if self.gamma is -1 else self.gamma[1]
        
        rangeGamma = np.linspace(start, end, self.shape[2])
        
        return rangeGamma[self.j - 1]
    
    def hitungPsi(self):
        start = (self.i + 1) if self.psi is -1 else self.psi[0]
        end = (self.i + self.j) if self.psi is -1 else self.psi[1]
        
        rangePsi = np.linspace(start, end, self.shape[2])
        
        return rangePsi[self.j - 1]

# Mendapatkan filter gabor dalam jumlah yang besar
# untuk digunakan sebagai kernel pada CNN
class GaborFilterBanks(Initializer):

    def __init__(self, gabor_params):
        self.gabor_params = gabor_params
        
    def getFilterBanks(self, shape):
        all_kernels = []
    
        # Membuat sejumlah kernel yang diinginkan
        for i in range(shape[3]):
            kernels = []
    
            # Membuat sejumlah channel
            for j in range(shape[2]):
                lambd, theta, psi, sigma, gamma = self.gabor_params.getParams(shape, i, j)
                
                kernels.append(
                    gaborFilter(
                        size = (shape[0], shape[1]), 
                        sigma = sigma,
                        theta = theta,
                        lambd = lambd,
                        gamma = gamma,
                        psi = psi
                    )
                )
                
            all_kernels.append(np.array(kernels))
        
        all_kernels = np.array(all_kernels).T
        
        return all_kernels

    def __call__(self, shape, dtype = None):
        all_kernels = self.getFilterBanks(shape)
        
        kernel = K.variable(all_kernels, dtype = dtype)

        return kernel
    