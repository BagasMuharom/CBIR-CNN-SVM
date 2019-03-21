import numpy as np

class PCA:
    
    __components = None
    
    __dataset = []
    
    'Mean dari setiap atribut data'
    __mean_vector = []
    
    'Nilai eigen'
    __nilai_eigen = None
    
    'Vektor eigen'
    __vector_eigen = None
    
    __kovarian = None
    
    __nilai_eigen = None
    
    __vektor_eigen = None
    
    __print = True
    
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
        
    def fit(self, mtx):
        self.__dataset = mtx
        self.__hitungMeanVector()
        self.__hitungZeroMean()
        self.__hitungKovarianMatriks()
        self.__hitungNilaiEigen()
        self.__pasangkanNilaiDanVektorEigen()
    
    def transform(self, data):
        return np.dot(data, self.__p_components)
    
    def __hitungMeanVector(self):
        for i in range(self.__dataset.shape[1]):
            self.__mean_vector.append(np.mean(self.__dataset[:, i]))
            
        self.__mean_vector = np.array(self.__mean_vector)
        
        if self.__print == True:
            print('--- Mean Vector ---')
            print(self.__mean_vector)
    
    def __hitungZeroMean(self):
        zeroMean = self.__dataset.copy().astype('float32')
        
        for i, mean in enumerate(self.__mean_vector):
            zeroMean[:, i] -= mean
            
        self.__zero_mean = zeroMean
            
        if self.__print == True:
            print('--- Zero Mean ---')
            print(zeroMean)
        
    def __hitungKovarianMatriks(self):
        self.__kovarian = self.__zero_mean.T.dot(self.__zero_mean)
        
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