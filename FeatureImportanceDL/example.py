import inspect
import sys
import os


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K

from loading_and_cleaning import load_dataset
from src.FeatureSelector import FeatureSelector
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#loading data
X = sys.argv[1]
Y = sys.argv[2]

X, Y = load_dataset.load_data(X, Y)

X_tr, X_te, y_tr ,y_te = train_test_split(X,Y,test_size=0.1,random_state=10)
X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=0.11,random_state=10)



# Training parapmeters
data_batch_size = 32
mask_batch_size = 32
# final batch_size is data_batch_size x mask_batch_size

s = X_tr.shape[0]
# size of optimal subset that we are looking for or the size of the snps that we need
#we coudl use percentages for this from the total number of features.

s_p = 2  # number of flipped bits in a mask when looking around m_opt
phase_2_start = 6000  # after how many batches phase 2 will begin
max_batches = 15000  # how many batches if the early stopping condition not satisfied
early_stopping_patience = 600  # how many patience batches (after phase 2 starts)
# before the training stops



# Dataset parameters
N_TRAIN_SAMPLES = X_tr.shape[0]
N_VAL_SAMPLES = X_val.shape[0]
N_TEST_SAMPLES = X_te.shape[0]
N_FEATURES = X.shape[1]
FEATURE_SHAPE = X.shape[1]
dataset_label = "GNP_"


print(X_tr.shape,X_val.shape,X_te.shape)
print(y_tr.shape,y_val.shape,y_te.shape)



# Create the framework, needs number of features and batch_sizes, str_id for tensorboard
fs = FeatureSelector(FEATURE_SHAPE, s, data_batch_size, mask_batch_size, str_id=dataset_label)

# Create a dense operator net, uses the architecture:
# N_FEATURES x 2 -> 60 -> 30 -> 20 -> 4
# with sigmoid activation in the final layer.
fs.create_dense_operator([60, 30, 20, 1], "softmax", metrics=[keras.metrics.CategoricalAccuracy()],
                       error_func=K.categorical_crossentropy)
# Ealy stopping activate after the phase2 of the training starts.
fs.operator.set_early_stopping_params(phase_2_start, patience_batches=early_stopping_patience, minimize=True)

# Create a dense selector net, uses the architecture:
# N_FEATURES -> 60 -> 30 -> 20 -> 4
fs.create_dense_selector([100, 50, 10, 1])

# Set when the phase2 starts, what is the number of flipped bits when perturbin masks
fs.create_mask_optimizer(epoch_condition=phase_2_start, perturbation_size=s_p)

#Train networks and set the maximum number of iterations
fs.train_networks_on_data(X_tr, y_tr, max_batches, val_data=(X_val, y_val))

#Results


#changing the feature importances to false
importances, optimal_mask = fs.get_importances(return_chosen_features=True)
optimal_subset = np.nonzero(optimal_mask)
test_performance = fs.operator.test_one(X_te, optimal_mask[None,:], y_te)
print("Importances: ", importances,len(importances) )

print("Optimal_subset: ", optimal_subset,len(optimal_subset))

print("Test performance (CE): ", test_performance[0])
print("Test performance (ACC): ", test_performance[1])