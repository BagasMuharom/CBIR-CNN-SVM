'''
Written by @srinadhu on Nov 19th.

reference: http://cs229.stanford.edu/materials/smo.pdf

'''

import numpy as np
import math
import random
import copy
from sklearn.metrics import accuracy_score
import sys
import time
from .Kernel import LinearKernel

def progressBar(current, total):
    bar_length = 40
    progress = current / total    
    block = int(round(bar_length * progress))
    text = "Validating: [{0}] {1:.1f}%".format( "=" * block + ">" + "." * (bar_length - block), progress * 100)
    sys.stdout.write('\r' + text)

class Callback():

    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

# Kelas untuk mengatur output berupa progress dari SVM
class ProgressManager():

    def __init__(self, max_passes, total_data = None, max_iter = -1):
        self.__iteration = 0
        self.__time = None
        self.__max_passes = max_passes
        self.__max_iter = max_iter
        self.__total_data = total_data

    def updateProgress(self, current_passes, current_iteration , time, index_data):
        progress_passes = current_passes / self.__max_passes
        progress_iteration = current_iteration / self.__max_iter
        
        text = '\rPasses: {0}/{1} | '.format(current_passes, self.__max_passes)

        if self.__max_iter > -1:
            text += 'Iteration: {0}/{1} | '.format(current_iteration, self.__max_iter)
        else:
            text += 'Iteration: {0}/~ | '.format(current_iteration)

        text += 'Index Data: {0}/{1} | Time: {2:.2f}s'.format( 
                index_data, 
                self.__total_data, 
                time
            )

        sys.stdout.write(text)

    def setTotalData(self, total_data):
        self.__total_data = total_data

class SVM():

    def __init__(self, kernel = LinearKernel(), tol = 0.001, C = 0.05):
        self.__kernel = kernel
        self.__weight = {}
        self.__bias = {}
        self.__C = C
        self.__tol = tol
        self.__X_val = None
        self.__Y_val = None

    def compile(self, callbacks = None, max_iter = -1, max_passes = -1, verbose = 1):
        self.__verbose = verbose
        self.__max_passes = max_passes
        self.__max_iter = max_iter

    def fit(self, X, Y, val_data = None):
        self.__X = X
        self.__Y = Y
        self.__initClassesList()

        if not val_data:
            self.__X_val = val_data[0]
            self.__Y_val = val_data[1]

        # Melakukan fit secara binary pada kelas ke-i dan kelas ke-j
        for i in range(self.__classes.shape[0]):
            self.__weight[i] = {}
            self.__bias[i] = {}
            for j in range(i + 1, self.__classes.shape[0]):

                weight, bias, alpha = self.fitBinary(self.__classes[i], self.__classes[j])
                self.__weight[i][j] = weight
                self.__bias[i][j] = bias

        return self

    def fitBinary(self, i, j):
        X_train, Y_train = self.__getXYBinary(self.__X, self.__Y, i, j)
        val_data = None

        if not self.__X_val and not self.__Y_val:
            X_val, Y_val = self.__getXYBinary(self.__X_val, self.__Y_val, i, j)
            val_data = (X_val, Y_val)

        smo = SMO(
                kernel = self.__kernel, 
                max_passes = self.__max_passes, 
                tol = self.__tol, 
                C = self.__C,
                max_iter = self.__max_iter
                )

        smo.compile(alias = {-1: i, 1: j})
        smo.fit(X_train, Y_train, val_data = val_data)

        return smo.getParams()

    def predict(self, X):

        y_pred = []

        for data in X:
            win = np.zeros(self.__classes.shape[0])

            for i in range(self.__classes.shape[0]):
                for j in range(i + 1, self.__classes.shape[0]):
                    weight = self.__weight[i][j]
                    bias = self.__bias[i][j]

                    result = np.sum(data * weight) + bias
                    win[i if result < 0 else j] += 1

            y_pred.append(self.__classes[np.argmax(win)])

        return y_pred

    def evaluate(self, X, Y):
        y_pred = self.predict(X)

        return accuracy_score(Y, y_pred)

    def summary(self):
        total_svm = self.__classes.shape[0] * (self.__classes.shape[0] - 1) / 2
        weight_size = self.__X.shape[1]
        weight_total = total_svm * weight_size
        bias_total = total_svm
        print('\n{0}'.format('=' * 80))
        print('Total Classes\t: ' + str(self.__classes.shape[0]))
        print('{0}'.format('=' * 80))
        print('Total SVM\t: ' + str(total_svm))
        print('{0}'.format('=' * 80))
        print('Total Weights\t: ' + str(weight_total))
        print('{0}'.format('=' * 80))
        print('Total Params\t: ' + str(weight_total + total_svm))
        print('{0}'.format('=' * 80))

    def __getXYBinary(self, X, Y, i, j):
        index_class_i = np.where(Y == i)
        index_class_j = np.where(Y == j)

        Xi = X[index_class_i]
        Xj = X[index_class_j]

        Yi = np.ones(Y[index_class_i].shape[0]) - 2
        Yj = np.ones(Y[index_class_j].shape[0])

        X_data = np.concatenate([Xi, Xj])
        Y_data = np.concatenate([Yi, Yj])

        return X_data, Y_data

    'Mendapatkan daftar kelas yang ada pada dataset'
    def __initClassesList(self):
        self.__classes = np.unique(self.__Y)

class SMO():

    def __init__(self, kernel, C = 0.05, tol = math.pow(10,-3), max_passes = -1, batch_size = None, verbose = 1, max_iter = -1):
        self.__kernel = kernel
        self.__C = C
        self.__tol = tol
        self.__max_passes = max_passes
        self.__weight = None
        self.__alpha = None
        self.__bias = None
        self.__X = None
        self.__Y = None
        self.__progress_manager = ProgressManager(
            max_passes=max_passes,
            max_iter=max_iter
        )
        self.__max_iter = max_iter
        self.__kernel_matriks = {}
        self.__batch_size = batch_size
        self.__verbose = verbose
        self.__alias = {
            1: 1,
            -1: -1
        }

    def compile(self, alias = {1: 1, -1: -1}):
        self.__alias = alias

    def getParams(self):
        return self.__weight, self.__bias, self.__alpha

    def fit(self, X, Y, val_data = None):

        self.__X = X
        self.__Y = Y

        if not val_data:
            self.__X_val = val_data[0]
            self.__Y_val = val_data[1]

        def printBeforeFit():
            if self.__verbose is not 1:
                return
            
            sys.stdout.write('\nFitting for class ' + str(self.__alias[-1]) + ' and ' + str(self.__alias[1]))
            sys.stdout.write('\nTrain on ' + str(X.shape[0]) + ' samples')

            if not val_data:
                sys.stdout.write(', validate on ' + str(self.__X_val.shape[0]) + ' samples')

        def printAccuracyInfo():
            if self.__verbose is not 1:
                return

            y_pred = self.predict(self.__X)
            y_true = self.__Y
            acc = accuracy_score(y_true, y_pred)

            if not val_data:
                val_acc = self.evaluate(self.__X_val, self.__Y_val)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            sys.stdout.write('acc: {0:.2f}'.format(acc))

            if not val_data:
                sys.stdout.write(' - val_acc: {0:.2f}'.format(val_acc))

        def printAfterFit():
            if self.__verbose is not 1:
                return
            printAccuracyInfo()
            sys.stdout.write('\n{0}'.format('-' * 80))

        self.__progress_manager.setTotalData(X.shape[0])
        
        printBeforeFit()

        self.__alpha, self.__bias = self.__countAlphaAndBias()
        self.__weight = self.__countWeight(self.__alpha)
            
        printAfterFit()

    def predict(self, X):
        y_pred = []

        for i in range(X.shape[0]):
            pred = np.sum(np.multiply(self.__weight, X[i, :])) + self.__bias

            if pred < 0:
                y_pred.append(-1)
            else:
                y_pred.append(1)

        return y_pred

    def evaluate(self, X, Y):
        sys.stdout.write('\nValidating ...')
        y_pred = []

        for i in range(X.shape[0]):
            pred = np.sum(np.multiply(self.__weight, X[i, :])) + self.__bias

            if pred < 0:
                y_pred.append(-1)
            else:
                y_pred.append(1)
            
            progressBar(i, X.shape[0])

        progressBar(X.shape[0], X.shape[0])

        return accuracy_score(Y, y_pred)

    def __countAlphaAndBias(self):
        start_time = time.time()
        alpha = np.zeros(shape=(self.__X.shape[0],1)) # each alpha[i] for every example.
        alpha_old = np.zeros(shape=(self.__X.shape[0], 1))
        b = 0
        E = np.zeros(shape = (self.__X.shape[0], 1)) #will be used in the loop 

        def countB(i, j):

            if (alpha[i] > 0 and alpha[i] < self.__C):
                ii = getKernelMatrix(i, i)
                ij = getKernelMatrix(i, j)
                b1 = b - E[i] - (self.__Y[i] * ii * (alpha[i] - alpha_old[i])) - (self.__Y[j] * ij * (alpha[j] - alpha_old[j]))

                return b1
            elif (alpha[j] > 0 and alpha[j] < self.__C):
                ij = getKernelMatrix(i, j)
                jj = getKernelMatrix(j, j)
                b2 = b - E[j] - (self.__Y[i] * ij * (alpha[i] - alpha_old[i])) - (self.__Y[j] * jj * (alpha[j] - alpha_old[j]))

                return b2
            else:
                ii = getKernelMatrix(i, i)
                ij = getKernelMatrix(i, j)
                jj = getKernelMatrix(j, j)
                delta_a_i = alpha[i] - alpha_old[i]
                delta_a_j = alpha[j] - alpha_old[j]
                b1_b2 = (2 * b) - E[i] -E[j] - (self.__Y[i] * ii * delta_a_i) - (self.__Y[j] * ij * delta_a_j) - (self.__Y[i] * ij * delta_a_i) - (self.__Y[j] * jj * delta_a_j)

                return b1_b2 / 2.0

        def updateAlphaJ(alpha_old_j, eta, L, H, i, j):
            alpha_j = alpha_old_j - ((self.__Y[j] * (E[i] - E[j])) / eta)

            if (alpha_j > H):
                alpha_j = H
            elif (alpha_j < L):
                alpha_j = L

            return alpha_j

        def countLH(i, j):
            if (self.__Y[i] != self.__Y[j]):
                L = max(0, alpha[j] - alpha[i])                        
                H = min(self.__C, self.__C + alpha[j] - alpha[i])

            else:
                L = max(0, alpha[i] + alpha[j] - self.__C)
                H = min(self.__C, alpha[i] + alpha[j])

            return L, H

        def countEta(i, j):
            eta = 2 * getKernelMatrix(i, j) 
            eta = eta - getKernelMatrix(i, i)
            eta = eta - getKernelMatrix(j, j)

            return eta

        def getKernelMatrix(i, j):
            result = None

            l = min(i, j)
            h = max(i, j)

            if l in self.__kernel_matriks:
                if h in self.__kernel_matriks[l]:
                    result = self.__kernel_matriks[l][h]
                else:
                    self.__kernel_matriks[l][h] = self.__kernel.transform(self.__X[l, :] , self.__X[h, :])
                    result = self.__kernel_matriks[l][h]
            else:
                self.__kernel_matriks[l] = {}
                self.__kernel_matriks[l][h] = self.__kernel.transform(self.__X[l, :] , self.__X[h, :])
                result = self.__kernel_matriks[l][h]

            return result

        def predict(index):
            result = 0

            for i in range(self.__X.shape[0]):
                kernel_matrik = getKernelMatrix(i, index)

                result += (alpha[i] * self.__Y[i] * kernel_matrik)

            result += b

            return result

        def printBeforeCount():
            if self.__verbose is not 1:
                return

            sys.stdout.write('\nStart Fitting ...')

        def printEachIteration(passes, iteration, index_data):
            self.__progress_manager.updateProgress(
                current_passes=passes, 
                current_iteration=iteration, 
                index_data=index_data, 
                time=time.time() - start_time
                )

        passes = 0
        iteration = 0

        printBeforeCount()

        while(passes < self.__max_passes and (iteration < self.__max_iter or self.__max_iter is -1)):

            iteration += 1
            num_changed_alphas = 0
            
            for i in range(self.__X.shape[0]): #for every example
                printEachIteration(passes=passes, iteration=iteration, index_data=i)
                
                E[i] = (predict(i) - self.__Y[i])
                    
                if ( (self.__Y[i] * E[i] < -self.__tol and alpha[i] < self.__C) or (self.__Y[i] * E[i] > self.__tol and alpha[i] > 0) ):
                        
                    j = i
                    
                    while(j == i):
                        j = random.randrange(self.__X.shape[0])
                        
                    E[j] = (predict(j) - self.__Y[j])

                    alpha_old[i] = alpha[i]
                    alpha_old[j] = alpha[j]
                    
                    #computing L and h values

                    L, H = countLH(i, j)

                    if (L==H):
                        continue

                    eta = countEta(i, j)
                
                    if (eta >= 0):
                        continue
                
                    #clipping

                    alpha[j] = updateAlphaJ(alpha_old[j], eta, L, H, i, j)

                    if (abs(alpha[j] - alpha_old[j]) < self.__tol):
                        continue
                
                    alpha[i] += (self.__Y[i] * self.__Y[j] * (alpha_old[j] - alpha[j])) #both alphas are updated

                    b = countB(i, j)

                    num_changed_alphas += 1
                #ended if
            #ended for
            if (num_changed_alphas == 0):
                passes+=1
            else:
                passes=0
        #end while

        printEachIteration(passes, iteration, self.__X.shape[0])

        return alpha, b   #returning the lagrange multipliers and bias.

    def __countWeight(self, alpha):
        weight = np.zeros(self.__X.shape[1])

        for i in range(self.__X.shape[0]):
            weight = np.add(weight, np.multiply(alpha[i] * self.__Y[i], self.__X[i, :]))

        return weight
