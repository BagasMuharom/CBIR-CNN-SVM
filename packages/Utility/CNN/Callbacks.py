from keras.callbacks import Callback
from os.path import join, isdir, exists
from os import mkdir

class ValidateModel(Callback):

    def __init__(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val
        self.loss = None
        self.val_acc = None
        self.history = {
            'loss' : [],
            'acc': []
        }
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs = None):
        self.loss, self.val_acc = self.model.evaluate(self.X_val, self.Y_val, verbose = 3)
        self.history['loss'].append(self.loss)
        self.history['acc'].append(self.val_acc)

        print('Accuracy :', self.val_acc)

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
        else:
            self.passes += 1
            print(f'Passes : {self.passes}/{self.maxpasses}')

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
        self.best_acc = 0
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

        if (self.save_best and val_acc > self.best_acc and val_acc > self.min_acc) or (val_acc >= self.min_acc):

            self.model.save('{0}/acc {1:.4f} - epoch {2}.h5'.format(self.output_dir, val_acc, epoch))

            print('Model Saved')

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                print(f'New best accuracy : {self.best_acc}')
            
