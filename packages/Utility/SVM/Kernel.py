import numpy as np
import math

class Kernel():

    def transform(self, x, y):
        raise NotImplementedError

# Kernel linear
class LinearKernel(Kernel):

    def transform(self, x, y):
        return np.sum(x * y)

# Kernel polynomial
class PolynomialKernel(Kernel):

    def __init__(self, degree = 1):
        self.__degree = degree

    def transform(self, x, y):
        dot_prdt = np.dot(np.transpose(x), y)

        return (dot_prdt + 1) ** self.__degree

# Kernel gaussian
class GaussianKernel(Kernel):

    def __init__(self, sigma = 0.5):
        self.__sigma = 0.5

    def transform(self, x, y):
        norm = np.linalg.norm(np.subtract(x, y)) #norm 

        res = math.exp(-(norm ** 2) / (2 * (self.__sigma ** 2))) #returning the final dot product.

        return res

class RadialBasisKernel(Kernel):

    def __init__(self, sigma):
        self.__sigma = sigma

    def transform(self, x, y):
        return np.sum(np.exp(np.abs(x - y) ** 2 / (2 * self.__sigma ** 2)))
