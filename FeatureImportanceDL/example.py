import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from src.FeatureSelector import FeatureSelector
from DataGenerator import generate_data, get_one_hot
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Dataset parameters
# N_TRAIN_SAMPLES = 512
# N_VAL_SAMPLES = 256
# N_TEST_SAMPLES = 1024
# N_FEATURES = 1002
# FEATURE_SHAPE = (10,)
# dataset_label = "GNP_"

# Training parapmeters
data_batch_size = 32
mask_batch_size = 32
# final batch_size is data_batch_size x mask_batch_size
s = 100  # size of optimal subset that we are looking for
s_p = 2  # number of flipped bits in a mask when looking around m_opt
phase_2_start = 6000  # after how many batches phase 2 will begin
max_batches = 15000  # how many batches if the early stopping condition not satisfied
early_stopping_patience = 600  # how many patience batches (after phase 2 starts)
# before the training stops

# Generate data for XOR dataset:


# In total 10 features
genotype_generated = pd.read_csv('Xsubset.csv', header=None)
phenotype_generated = pd.read_csv('hapmap_phenotype_recoded',header=None)

genotype_generated = genotype_generated.to_numpy()
phenotype_generated = phenotype_generated.to_numpy()
phenotype_generated = phenotype_generated.reshape(len(phenotype_generated),)


X_tr, X_te, y_tr ,y_te = train_test_split(genotype_generated,phenotype_generated,test_size=0.1,random_state=10)

X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=0.11,random_state=10)

# Dataset parameters
N_TRAIN_SAMPLES = X_tr.shape[0]
N_VAL_SAMPLES = X_val.shape[0]
N_TEST_SAMPLES = X_te.shape[0]
N_FEATURES = 1002
FEATURE_SHAPE = (1002,)
dataset_label = "GNP_"


print(X_tr.shape,X_val.shape,X_te.shape)
print(y_tr.shape,y_val.shape,y_te.shape)

# # Get one hot encoding of the labels
# y_tr = get_one_hot(y_tr.astype(np.int8), 4)
# y_te = get_one_hot(y_te.astype(np.int8), 4)
# y_val = get_one_hot(y_val.astype(np.int8), 4)

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

print("Optimal_subset: ", optimal_subset)
header_file = pd.read_csv('snps_list_shorter.txt')
print("Test performance (CE): ", test_performance[0])
print("Test performance (ACC): ", test_performance[1])