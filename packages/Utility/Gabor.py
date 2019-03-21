import numpy as np
import keras.backend as K
from keras.initializers import Initializer
import cv2 as cv

# Membuat filter gabor
def gaborFilter(size, lambd, theta, psi, sigma, gamma):
    gabor = np.zeros(shape = size)
    half_height = int(size[1] / 2)
    half_width = int(size[0] / 2)
    
    for y in range(-half_height, half_height + 1):
        for x in range(-half_width, half_width + 1):
            x_aksen = x * np.cos(theta) + y * np.sin(theta)
            y_aksen = -x * np.sin(theta) + y * np.cos(theta)
            gabor[y + half_height][x + half_width] = np.exp(-(x_aksen ** 2 + ((gamma ** 2) * (y_aksen ** 2))) / (2 * (sigma ** 2))) * np.cos((((2 * np.pi * x_aksen) / lambd) + psi))

    return gabor

# Mendapatkan filter gabor dalam jumlah yang besar
# untuk digunakan sebagai kernel pada CNN
class GaborFilterBanks(Initializer):

    def __init__(self, lambd = 10, theta = 0, psi = 1.57, sigma = 5, gamma = 0.75):
        self.__lambd = lambd
        self.__theta = theta
        self.__psi = psi
        self.__sigma = sigma
        self.__gamma = gamma

    def __call__(self, shape, dtype = None):
        all_kernels = []
    
        # Membuat sejumlah kernel yang diinginkan
        for i in range(shape[3]):
            kernels = []
            
            # Membuat sejumlah channel
            for j in range(shape[2]):
                kernels.append(
                    gaborFilter(
                        size = (shape[0], shape[1]), 
                        sigma = self.__sigma,
                        theta = i * np.pi / 180, # Ubah dari derajat ke radian
                        lambd = self.__lambd,
                        gamma = self.__gamma,
                        psi = self.__psi
                    )
                )
                
            all_kernels.append(np.array(kernels))
        
        all_kernels = np.array(all_kernels).T
        
        kernel = K.variable(all_kernels, dtype = dtype)

        return kernel

    def get_config(self):
        return {
            'lambd': self.__lambd,
            'psi': self.__psi,
            'sigma': self.__sigma,
            'theta': self.__theta,
            'gamma': self.__gamma
        }
    