import numpy as np

class PCA2D:
    
    def __init__(self, components, print = True):
        self.__components = components
        self.__dataset = None
        self.__mean_vector = []
        self.__nilai_eigen = None
        self.__vector_eigen = None
        self.__kovarian = None
        self.__nilai_eigen = None
        self.__vektor_eigen = None
        self.__print = print
        self._p_components = None
        
    def fit(self, images):
        self.__dataset = images
        self.__hitungMeanVector()
        self.__hitungZeroMean()
        self.__hitungKovarianMatriks()
        self.__hitungNilaiEigen()
        self.__pasangkanNilaiDanVektorEigen()
    
    def transform(self, data):
        transformed = []
        
        for i in range(data.shape[0]):
            transformed.append(np.dot(data[i], self.__p_components).T)
        
        return np.array(transformed).T
    
    def __hitungMeanVector(self):
        self.__mean_vector = np.mean(self.__dataset, axis = 0)
        
        if self.__print == True:
            print('--- Mean Vector ---')
            print(self.__mean_vector)
    
    def __hitungZeroMean(self):
        self.__zero_mean = self.__dataset - self.__mean_vector
            
        if self.__print == True:
            print('--- Zero Mean ---')
            print(self.__zero_mean)
        
    def __hitungKovarianMatriks(self):
        kovarian_individu = []
        
        for i in range(self.__dataset.shape[0]):
            transpose = (self.__dataset[i] - self.__mean_vector).T
            kovarian_individu.append(transpose.dot(self.__zero_mean[i]))
        
        self.__kovarian = (1 / (self.__dataset.shape[0] - 1)) * np.sum(kovarian_individu, axis = 0)
        
        if self.__print == True:
            print('--- Kovarian ---')
            print(self.__kovarian)
        
    def __hitungNilaiEigen(self):
        self.__nilai_eigen, self.__vektor_eigen = np.linalg.eig(self.__kovarian)
        
        if self.__print == True:
            print('--- Nilai Eigen ---')
            print(self.__nilai_eigen)
            
            print('--- Vektor Eigen ---')
            print(self.__vektor_eigen)
        
    def __pasangkanNilaiDanVektorEigen(self):
        self.__pasangan_eigen = [[np.abs(self.__nilai_eigen[i]), self.__vektor_eigen[:,i]] for i in range(len(self.__nilai_eigen))]
        
        self.__pasangan_eigen.sort(key=lambda x: x[0], reverse=True)
        
        pc = []
        
        for i, item in enumerate(self.__pasangan_eigen):
            pc.append(item[1].reshape(self.__dataset.shape[1], 1))
                
            if i == self.__components - 1:
                break
            
        self.__p_components = np.hstack(pc)
        
        if self.__print == True:
            print('--- Pasangan Eigen ---')
            print(self.__pasangan_eigen)

            print('--- Components ---')
            print(self.__p_components)