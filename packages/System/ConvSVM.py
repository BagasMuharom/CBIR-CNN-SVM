import numpy as np

class ConvSVM:
    
    def __init__(self):
        self.__model = None
        self.__svm = None
        self.__dataset = None
        self.__extracted = None
        self.__xPos = None
        self.__yPos = None
        self.__index = None
        
    def setModel(self, model):
        self.__model = model.getModel()
        
    def setSVM(self, svm):
        self.__svm = svm
        
    def setDataset(self, dataset, xPos, yPos):
        self.__dataset = dataset
        self.__xPos = xPos
        self.__yPos = yPos
        
    def fit(self, extractFeature = True):
        if extractFeature:
            self.extractFeature()
            
        self.fitSVM()
        
    def setIndex(self, index):
        self.__index = index
        
    def setFeature(self, feature):
        self.__extracted = feature
        
    def getFeature(self):
        return self.__extracted
        
    def extractFeature(self):
        print('--------- extracting')
        self.__extracted = self.__model.predict(np.array([i[self.__xPos] for i in self.__dataset]))
        
        return self.__extracted
    
    def fitSVM(self):
        print('----------- fitting')
        Y = None
        X = self.__extracted
        
        if self.__index is None:
            Y = self.__dataset[:, self.__yPos].astype('int')
            
        else:
            Y = self.__dataset[self.__index, self.__yPos].astype('int')
            X = self.__extracted[self.__index]
            
        self.__svm.fit(X, Y)
        print('---------- fitting complete')
        
    'dataset => merupakan array berisi N x M x D yang menggambarkan sebuah citra'
    def predict(self, dataset):
        print('-------- predicting')
        extracted = self.__model.predict(np.array([i[0] for i in dataset]))

        return self.__svm.predict(extracted)
    
    def predictFromFeature(self, features):
        return self.__svm.predict(features)
    