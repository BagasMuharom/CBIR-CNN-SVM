import numpy as np
import os
from os import listdir
from os.path import isdir, isfile, join
import cv2 as cv
from keras.utils import to_categorical
from sklearn.model_selection import KFold

class DatasetWriter:

    def __init__(self, path, output_dir, jumlah_citra_per_kelas, fold_split, resize = False, shuffle = True, grayscale = False):
        self.path = path
        self.output_dir = output_dir
        self.jumlah_citra_per_kelas = jumlah_citra_per_kelas
        self.resize = resize
        self.shuffle = shuffle
        self.grayscale = grayscale
        self.dataset = []
        self.folds = []
        self.fold_split = fold_split
        self.daftar_direktori = []

    def initDataset(self):
        self.daftar_direktori = [direktori for direktori in os.listdir(self.path) if isdir(join(self.path, direktori))]

        for index_direktori, direktori in enumerate(self.daftar_direktori):
            daftar_citra = [join(join(self.path, direktori), citra) for citra in os.listdir(join(self.path, direktori)) if isfile(join(join(self.path, direktori), citra))]
            
            if self.shuffle:
                daftar_citra = np.random.choice(daftar_citra, self.jumlah_citra_per_kelas, False)
            
            maks_citra = range(len(daftar_citra))
            
            if self.jumlah_citra_per_kelas > 0:
                maks_citra = range(0, self.jumlah_citra_per_kelas)
            
            for i in maks_citra:
                citra = cv.imread(daftar_citra[i])
                
                if self.resize is not False:
                    citra = cv.resize(citra, self.resize)
                    
                if self.grayscale:
                    citra = cv.cvtColor(citra, cv.COLOR_BGR2GRAY)
                
                target = index_direktori
                target_categorical = to_categorical(target, len(self.daftar_direktori))
                
                dataset.append(
                    np.array([
                        citra, target, target_categorical, daftar_citra[i]
                    ])
                )
                
        dataset = np.array(dataset)

    def initFold(self):
        jumlah_kelas = len(self.daftar_direktori)
        jumlah_citra_per_kelas = self.jumlah_citra_per_kelas
        kf = KFold(n_splits = self.fold_split, shuffle = True)
        folds = []

        for i in range(self.fold_split):
            folds.append([[],[]])

        # Melakukan k-fold untuk setiap kelas
        for i in range(jumlah_kelas - 1):
            start_idx = i * jumlah_citra_per_kelas
            end_idx = (i + 1) * jumlah_citra_per_kelas
            
            fold_iter = 0
            for train, test in kf.split(self.dataset[start_idx:end_idx]):
                for tr in train:
                    folds[fold_iter][0].append(int(tr + start_idx))
                
                for te in test:
                    folds[fold_iter][1].append(int(te + start_idx))
                fold_iter += 1
            
        self.folds = np.array(folds)

    def save(self):
        np.save(join(self.output_dir, 'dataset.npy'), self.dataset)
        np.save(join(self.output_dir, 'folds.npy'), self.folds)
        np.save(join(self.output_dir, 'target.npy'), self.daftar_direktori)

class DatasetReader:

    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.folds = []
        self.labels = []

    def getDataset(self):
        self.dataset = np.load(join(self.path, 'dataset.npy'))
        self.folds = np.load(join(self.path, 'folds.npy'))
        self.labels = np.load(join(self.path, 'labels.npy'))

    def getTrainData(self, fold):
        X_train = np.array([i[0] for i in self.dataset[self.folds[fold, 0]]])
        Y_train_onehot = np.array([i[2] for i in self.dataset[self.folds[fold, 0]]])
        Y_train_single = np.array([i[1] for i in self.dataset[self.folds[fold, 0]]])

    def getTestData(self, fold):
        X_test = np.array([i[0] for i in self.dataset[self.folds[fold, 1]]])
        Y_test_onehot = np.array([i[2] for i in self.dataset[self.folds[fold, 1]]])
        Y_test_single = np.array([i[1] for i in self.dataset[self.folds[fold, 1]]])
