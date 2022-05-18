import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow
import keras
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras import utils
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import batch_normalization
from keras.regularizers import l2
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, scale
from keras.utils import np_utils

def NN1(input_dim, output_dim, isClassification = True):
    print("Starting NN1")
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(100, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    if (isClassification == False):
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='sgd')
    elif (isClassification == True):
        model.add(Dense(output_dim, activation='softmax', kernel_initializer='normal'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


def NN2(input_dim, output_dim, isClassification=True):
    print("Starting NN2")
    model = Sequential()
    model.add(
        Dense(50, input_dim=input_dim, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))

    if (isClassification == False):
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='sgd')
    elif (isClassification == True):
        model.add(Dense(output_dim, activation='softmax', kernel_initializer='normal'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


# Deep Model
def DeepNN(input_dim, output_dim, isClassification=True):
    print("Starting DeepNN")
    model = Sequential()
    model.add(Dense(500, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    model.add(batch_normalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_initializer='normal'))
    model.add(batch_normalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, kernel_initializer='normal', kernel_regularizer=l2(0.1)))
    model.add(batch_normalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer='random_uniform', kernel_regularizer=l2(0.1)))
    model.add(batch_normalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, kernel_initializer='random_uniform', kernel_regularizer=l2(0.1)))
    model.add(batch_normalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_initializer='normal', kernel_regularizer=l2(0.1)))
    model.add(batch_normalization())
    model.add(Dropout(0.5))
    model.add(Dense(500, kernel_initializer='normal'))
    model.add(batch_normalization())
    model.add(Dropout(0.2))
    model.add(PReLU())

    if (isClassification == False):
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    elif (isClassification == True):
        model.add(Dense(output_dim, activation='softmax', kernel_initializer='normal'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# VIANN
# Variance-based Feature Importance of Artificial Neural Networks
class VarImpVIANN(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0

    def on_train_begin(self, logs={}, verbose=1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.diff = self.model.layers[0].get_weights()[0]

    def on_epoch_end(self, batch, logs={}):
        currentWeights = self.model.layers[0].get_weights()[0]

        self.n += 1
        delta = np.subtract(currentWeights, self.diff)
        self.diff += delta / self.n
        delta2 = np.subtract(currentWeights, self.diff)
        self.M2 += delta * delta2

        self.lastweights = self.model.layers[0].get_weights()[0]

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float('nan')
        else:
            self.s2 = self.M2 / (self.n - 1)

        scores = np.sum(np.multiply(self.s2, np.abs(self.lastweights)), axis=1)

        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        if self.verbose:
            print("Most important variables: ",
                  np.array(self.varScores).argsort()[-10:][::-1])

# Taken from https://csiu.github.io/blog/update/2017/03/29/day33.html
def garson(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    # weight through node (axis=0 is column; sum per input feature)
    cw_h = abs(cw).sum(axis=0)

    # relative contribution of input neuron to outgoing signal of each hidden neuron
    # sum to find relative contribution of input neuron
    rc = np.divide(abs(cw), abs(cw_h))
    rc = rc.sum(axis=1)

    # normalize to 100% for relative importance
    ri = rc / rc.sum()
    return(ri)


# Adapted from https://csiu.github.io/blog/update/2017/03/29/day33.html
class VarImpGarson(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def on_train_end(self, batch, logs={}):
        if self.verbose:
            print("VarImp Garson")
        """
        Computes Garson's algorithm
        A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
        B = vector of weights of hidden-output layer
        """
        A = self.model.layers[0].get_weights()[0]
        B = self.model.layers[len(self.model.layers) - 1].get_weights()[0]

        self.varScores = 0
        for i in range(B.shape[1]):
            self.varScores += garson(A, np.transpose(B)[i])
        if self.verbose:
            print("Most important variables: ",
                  np.array(self.varScores).argsort()[-10:][::-1])

# Leave-One-Feature-Out LOFO
def LeaveOneFeatureOut(model, X, Y):
    OneOutScore = []
    n = X.shape[0]
    for i in range(0,X.shape[1]):
        newX = X.copy()
        newX[:,i] = 0 #np.random.normal(0,1,n)
        OneOutScore.append(model.evaluate(newX, Y, batch_size=2048, verbose=0))
    OneOutScore = pd.DataFrame(OneOutScore[:])
    ordered = np.argsort(-OneOutScore.iloc[:,0])
    return(OneOutScore, ordered)

#Testing variable importance
#Settings obtained for each dataset
VIANN = VarImpVIANN(verbose=1)
input_dim = 1024

model = Sequential()
model.add(Dense(50, input_dim=input_dim, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))

model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))

model.add(Dense(50, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))

model.add(Dense(5, activation='softmax', kernel_initializer='normal'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.05, epochs=30, batch_size=64, shuffle=True,
          verbose=1, callbacks=[VIANN])

print(VIANN.varScores)
