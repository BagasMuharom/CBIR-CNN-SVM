import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

'''
Class ini berfungsi untuk menggambarkan dan menyimpan confusion matrix
pada suatu data. Confusion matrix yang ditampilkan bisa lebih dari satu
'''
class ConfusionMatrix:

    def __init__(self, figsize, num_rows, num_cols, labels):
        self.figsize = figsize
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.plots = []
        self.fig = None
        self.gs = None
        self.labels = labels
        self.xylabel_size = 12
        self.label_size = 12
        self.count_size = 12
        self.suptitle = None

    '''
    Menambahkan confusion matrix
    '''
    def add(self, Y_true, Y_pred, title, row_title = 'Kelas Sebenarnya', col_title = 'Kelas Prediksi', cmap = plt.cm.Blues):
        self.plots.append(
            [Y_true, Y_pred, title, row_title, col_title, cmap]
        )

    '''
    Menampilkan seluruh confusion matrix
    '''
    def show(self):
        self.fig = plt.figure(figsize = self.figsize, dpi = 300)
        self.gs = GridSpec(self.num_rows, self.num_cols, figure = self.fig)
        self.fig.suptitle(self.suptitle, fontsize = 20, va = 'center', ha = 'center')

        for index, subplot in enumerate(self.plots):
            self.draw_conf_mat(subplot, index + 1)

    '''
    Menggambar suatu confusion matrix
    '''
    def draw_conf_mat(self, data, index):
        Y_true = data[0]
        Y_pred = data[1]
        title = data[2]
        ylabel = data[3]
        xlabel = data[4]
        cmap = data[5]

        # Mendapatkan lokasi confusion matrix yang akan ditampilkan
        row_pos = int(np.ceil(index / self.num_cols))
        col_pos = int(index - (self.num_cols * (row_pos - 1)))

        conf_mat = confusion_matrix(Y_true, Y_pred)
        percentage = (conf_mat / conf_mat.sum(axis = 1)[:, np.newaxis]) * 100

        axes = plt.subplot(self.gs[row_pos - 1, col_pos - 1])
        im = axes.imshow(conf_mat, interpolation = 'nearest', cmap = cmap)

        axes.set(
            xticks = np.arange(conf_mat.shape[1]),
            yticks = np.arange(conf_mat.shape[0]),
            xticklabels = self.labels, 
            yticklabels = self.labels,
            ylabel = ylabel,
            xlabel = xlabel
            )
        
        axes.title.set_position([0.5, 1.1])
        axes.title.set_text(title)
        axes.title.set_size(15)
        axes.yaxis.label.set_size(self.xylabel_size)
        axes.xaxis.label.set_size(self.xylabel_size)

        # Rotate the tick labels and set their alignment.
        plt.setp(axes.get_xticklabels(), rotation = 45, ha = "right",
                rotation_mode = "anchor", fontsize = self.label_size)
        plt.setp(axes.get_yticklabels(), fontsize = self.label_size)

        # Loop over data dimensions and create text annotations.
        thresh = conf_mat.max() / 2.
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                content = f'{conf_mat[i, j]}\n{percentage[i, j]:.2f}%'

                axes.text(j, i, content,
                        ha="center", va="center",
                        color="white" if conf_mat[i, j] > thresh else "black",
                        fontsize = self.count_size)
                
        akurasi = accuracy_score(Y_true, Y_pred) * 100
        axes.text(0.5, 1.05, f'Akurasi : {akurasi:.2f}%', fontsize = 13, ha = 'center', transform = axes.transAxes)
        
    '''
    Menyimpan confusion matrix pada path tertentu
    '''
    def save(self, path, dpi = 400):
        plt.savefig(path, bbox_inches = 'tight', dpi = dpi)

class Evaluator:

    def __init__(self, model, labels):
        self.model = model

    def evaluate(self, X, Y):
        raise NotImplementedError

class CNN_Evaluator(Evaluator):

    def evaluate(self, X, Y):
        y_true = Y
        y_pred = np.argmax(
            self.model.predict(X), axis = 1
        )
        score = accuracy_score(y_true, y_pred)

        return y_true, y_pred, score

class SVM_Evaluator(Evaluator):

    def evaluate(self, X, Y):
        y_true = Y
        y_pred = self.model.predict(X)
        
        score = accuracy_score(y_true, y_pred)

        return y_true, y_pred, score
