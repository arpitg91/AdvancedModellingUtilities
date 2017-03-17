from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn.cross_validation import train_test_split

class NN:
    '''
    Make a wrapper of keras and theano to make it easy to use. The functions are written as in sklearn
    Parameters as on: http://keras.io/
    '''
    def __init__(self, inputShape, layers, dropout = [], activation = 'relu', init = 'uniform', loss = 'rmse', optimizer = 'adadelta', nb_epochs = 50, batch_size = 256, verbose = 1):

        model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                print ("Input shape: " + str(inputShape))
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], input_dim = inputShape, init = init))
            else:
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], init = init))
            print ("Adding " + activation + " layer")
            model.add(Activation(activation))
            model.add(BatchNormalization())
            if len(dropout) > i:
                print ("Adding " + str(dropout[i]) + " dropout")
                model.add(Dropout(dropout[i]))
        model.add(Dense(1, init = init)) #End in a single output node for regression style output
        model.compile(loss=loss, optimizer=optimizer)
        
        self.model = model
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y): 
        # print X.shape, np.array(y).reshape((len(y),1)).shape
        self.model.fit(np.array(X), np.array(y).reshape((len(y),1))-1, nb_epoch=self.nb_epochs, batch_size=self.batch_size, verbose = self.verbose)
        
    def predict(self, X, batch_size = 128, verbose = 1):
        return self.model.predict(np.array(X), batch_size = batch_size, verbose = verbose)+1

    def get_params(self, deep=True):
        return { 'layers':self.layers, 'dropout':self.dropout, 'activation':self.activation, 'init':self.init,\
                    'loss':self.loss, 'optimizer':self.optimizer, 'nb_epochs':self.nb_epochs, 'batch_size':self.batch_size, 'verbose':self.verbose}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
