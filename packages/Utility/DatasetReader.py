import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import cv2 as cv

class DatasetReader:

    def __init__(self):
        self.__resize = None
        self.__binarization = False

    def resize(self, size):
        self.__resize = size

    def binarization(self):
        self.__binarization = True

    # Mendapatkan daftar direktori dari sebuah direktori
    def getDirectoriesList(self, dir):
        return [folder for folder in listdir(dir) if isdir(join(dir, folder))]

    # Mendapatkan daftar citra dari sebuah direktori
    def getImagesList(self, dir):
        return [join(dir, image) for image in listdir(dir) if isfile(join(dir, image))]

    # Mendapatkan dataset dari sebuah direktori dengan syarat sebagai berikut :
    # 1. Direktori harus mengandung direktori dengan nama kelas yang mewakili citra pada direktori itu
    def getDataset(self, dir):
        # Untuk menyimpan dataset
        dataset = []
        
        directories = self.getDirectoriesList(dir)

        # Melakukan scanning direktori pada sebuah direktori
        for indexFolder, folder in enumerate(directories):
            # Melakukan scanning citra pada sebuah direktori
            for indexImage, image in enumerate(self.getImagesList(join(dir, folder))):
                # Menambahkan ke dalam dataset
                dataset.append([
                    self.__readImages(image), indexFolder, self.__defineTarget(len(directories), indexFolder), image
                ])

        return np.array(dataset)

    def __readImages(self, dir):
        img = cv.imread(dir)

        if self.__resize is not None:
            img = cv.resize(img, self.__resize)
    
            if self.__binarization:
                retval, threshold = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
                img = np.array(threshold[:, :, 0]).reshape((32, 32, 1))

        return img
    
    def __defineTarget(self, totalFolder, indexFolder):
        target = np.zeros(totalFolder)
        target[indexFolder] = 1
        
        return target