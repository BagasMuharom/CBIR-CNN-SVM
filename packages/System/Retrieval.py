import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class Retrieval:
    
    def __init(self, dataset, model):
        self.dataset = dataset
        self.model = model
        
    '''
    Melakukan proses retrieval atau pengembalian citra
    '''
    def retrieve(self, X, count = 20):
        y_pred = self.__classify(X)
        library = self.__getLibrary(y_pred)

        distance = self.__hitungJarak(X, library)

        return distance[:count]

    def showRetrievalResult(self, retrieved, query_dir, num_rows, num_cols, figsize = (20, 20)):
        fig = plt.figure(figsize = figsize)

        # Menampilkan citra kueri
        query = cv.imread(query_dir)
        query = cv.cvtColor(query, cv.COLOR_BGR2RGB)

        axes.add_subplot(num_rows, num_cols, 1)
        axes.imshow(query)
        axes.yticks([])
        axes.xticks([])

        # Menampilkan hasil retrieval
        for i in range(retrieved.shape[0]):
            image = cv.imread(retrieved[i, -1])
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            axes = fig.add_subplot(num_rows, num_cols, i + 2)
            axes.imshow(image)
            axes.yticks([])
            axes.xticks([])

        plt.show()


    '''
    Melakukan klasifikasi terhadap fitur kueri
    '''
    def __classify(self, X):
        y_pred = self.model.predict(np.array([X]))

        return y_pred[0]

    '''
    Mendapatkan daftar fitur sesuai kelas yang sama dengan hasil
    klasifikasi fitur kueri
    '''
    def __getLibrary(self, y_pred):
        return self.dataset[
            np.where(self.dataset[:, 1] == y_pred)
        ]

    '''
    Menghitung jarak terhadap setiap fitur pada perpustakaan fitur
    dengan fitur kueri
    '''
    def __hitungJarak(self, X, library):
        distance = []

        for i in range(library.shape[0]):
            distance.append(
                # Menambahkan jarak fitur
                euclidean(X, library[i, 0]),
                # Menambahkan direktori citra
                library[i, -1]
            )

        return distance.sort(key = lambda x: x[0])
