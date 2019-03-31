from keras.models import Model
import numpy as np

class FeatureExtractor:

    def __init__(self, model, output_layer = 'flatten'):
        self.model = model
        self.output = output_layer
        self.extractor_model = None
        self.defineExtractorModel()

    '''
    Mendefinisikan model baru dengan output layer yang berbeda.
    Umumnya output layer diarahkan ke flatten layer
    '''
    def defineExtractorModel(self):
        self.extractor_model = Model(
            inputs = self.model.input,
            outputs = self.model.get_layer(self.output).output
        )

    'Melakukan ekxtraksi fitur terhadap kumpulan data X'
    def extract(self, X):
        feature = self.extractor_model.predict(X)

        return feature

'''
Class ini berfungsi untuk menulis feature yang sudah didapatkan 
ke dalam file
'''
class FeatureWriter:

    def __init__(self, feature, dataset):
        pass

'''
Class ini berfungsi untuk membaca feature yang sudah disimpan
ke dalam file agar bisa digunakan kembali untuk proses
retrieval
'''
class FeatureReader:

    def __init__(self, path):
        self.path = path
        self.feature = []
        self.read()

    def read(self):
        self.feature = np.load(self.path)

        return self.feature
