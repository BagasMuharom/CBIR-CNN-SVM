from keras.callbacks import Callback

class SaveModel(Callback):
    
    def __init__(self, X_val, Y_val, base_dir, max_passes = 50, min_acc = 0.5):
        self.max_passes = max_passes
        self.min_acc = min_acc
        self.X_val = X_val
        self.Y_val = Y_val
        self.best_acc = 0
        self.passes = 0
        self.base_dir = base_dir
        
    def on_epoch_end(self, epoch, logs = None):
        val_acc = self.model.evaluate(self.X_val, self.Y_val, verbose=3)[1]
        print('val_acc: {:.3f}'.format(val_acc))
        # jika akurasi yang didapat lebih besar
        if val_acc > self.best_acc:
            self.passes = 0
            self.best_acc = val_acc
            
            if val_acc >= self.min_acc:
                self.model.save('{0}/acc {1:.4f} - epoch {2}.h5'.format(self.base_dir, val_acc, epoch))
        else:
            self.passes += 1
            print('Passes: {0}/{1}'.format(self.passes, self.max_passes))
            
        if self.passes == self.max_passes:
            self.model.stop_training = True