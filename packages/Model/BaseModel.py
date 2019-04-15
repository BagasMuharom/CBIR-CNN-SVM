class BaseModel:

    def __init__(self, kernels = {}, optimizer = 'adadelta', loss = 'categorical_crossentropy'):
        self.optimizer = optimizer
        self.loss = loss
        self.model = None
        self.kernels = kernels
        self.initModel()

    def initModel(self):
        raise NotImplementedError

    def setKernel(self, kernels):
        self.kernels = kernels

    def compile(self):
        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = ['accuracy']
        )
        
    def save(self, path):
        self.model.save(path)