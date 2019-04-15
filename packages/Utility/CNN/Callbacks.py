from keras.callbacks import Callback
from sklearn.externals import joblib
from os.path import join, isdir, exists
from os import mkdir

class ValidateModel(Callback):

    def __init__(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val
        self.loss = None
        self.val_acc = None
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs = None):
        self.loss, self.val_acc = self.model.evaluate(self.X_val, self.Y_val, verbose = 3)

        if self.val_acc > self.best_acc:
            self.best_acc = self.val_acc
            print(f'New best accuracy : {self.best_acc}')
        else:
            print('Accuracy :', self.val_acc)

class HistorySaver(Callback):

    def __init__(self, validate_model, base_dir, fold):
        self.val_model = validate_model
        self.path = f'{base_dir}/Fold {fold}/history.sav'
        self.history = {
            'loss' : [],
            'acc': []
        }

    def initHistory(self):
        if exists(path):
            self.history = joblib.load(self.path)

    def on_epoch_end(self, epoch, logs = None):
        self.history['loss'].append(self.val_model.loss)
        self.history['acc'].append(self.val_model.val_acc)

    def on_train_end(self, logs = None):
        save = joblib.dump(self.history, self.path)

        print('History saved')

class StopTraining(Callback):

    def __init__(self, validate_model, min_acc = -1, max_passes = 50):
        self.min_acc = min_acc
        self.max_passes = max_passes
        self.best_acc = 0
        self.passes = 0
        self.validate_model = validate_model

    def on_epoch_end(self, epoch, logs):
        if self.validate_model.val_acc > self.best_acc:
            self.best_acc = self.validate_model.val_acc
            self.passes = 0
        else:
            self.passes += 1

        print(f'Passes : {self.passes}/{self.max_passes}')
        
        if self.passes is self.max_passes:
            self.model.stop_training = True

        if self.validate_model.val_acc >= self.min_acc and self.min_acc is not -1:
            self.model.stop_training = True

class SaveModel(Callback):
    
    def __init__(self, validate_model, base_dir, fold, min_acc = 0.5, save_best = False):
        self.base_dir = base_dir
        self.fold = fold
        self.save_best = save_best
        self.validate_model = validate_model
        self.output_dir = None
        self.min_acc = min_acc
        self.initOutputDir()

    def initOutputDir(self):
        if not exists(join(self.base_dir, 'Fold ' + str(self.fold))):

            if not exists(self.base_dir):
                mkdir(self.base_dir)

            mkdir(join(self.base_dir, 'Fold ' + str(self.fold)))

        self.output_dir = join(self.base_dir, 'Fold ' + str(self.fold))
        
    def on_epoch_end(self, epoch, logs = None):
        val_acc = self.validate_model.val_acc

        if (self.save_best and val_acc > self.validate_model.best_acc and val_acc > self.min_acc) or (val_acc >= self.min_acc):

            self.model.save_weights('{0}/acc {1:.4f} - epoch {2}.h5'.format(self.output_dir, val_acc, epoch))

            print('Model saved to disk')
