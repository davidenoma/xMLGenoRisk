from sklearn import clone
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import  pandas as pd
from sklearn.preprocessing import LabelEncoder
# from matplotlib import pyplot as plt
#HYPER
# [250, 300]
n_estimators = 250
# [2, 4, 6, 8]
max_depth = 8
# [0.001, 0.01, 0.1]
learning_rate = 0.001

genotype_generated = pd.read_csv('../Xsubset.csv', header=None)
phenotype_generated = pd.read_csv('../hapmap_phenotype_recoded',header=None)


#Conversion to numpy
genotype_generated = genotype_generated.values.astype(np.int64)
phenotype_generated = phenotype_generated.values.astype(np.int64)
phenotype_generated = phenotype_generated.ravel()
#Conversion of phenotype
# phenotype_generated = LabelEncoder().fit_transform(phenotype_generated)



# Create a train/test/validation split
#the below two lines of code generate a 60, 20, 20 split
X_train, X_test, y_train, y_test = train_test_split(genotype_generated, phenotype_generated, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
#the below two lines of code to generate a 80 10 10 split.
# X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
# X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train ,y_test = train_test_split(genotype_generated,phenotype_generated,test_size=0.2)

model = XGBClassifier(nthread=4, seed=0, n_estimators=n_estimators, max_depth=max_depth,learning_rate=learning_rate)
eval_set = [(X_test, y_test)]
# Removing this parameter
# eval_set=eval_set
# early_stopping_rounds=model.n_estimators / 10
print(X_train.shape)
model = clone(model)

model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=model.n_estimators,verbose = True)
scores_key_values = model.get_booster().get_score(importance_type='gain')
# print(list(scores_key_values.values()),"\n", list(scores_key_values.keys()))
# plt.barh([i for i in range(0,X_train.shape[1])], model.feature_importances_)
# plt.show()
print(scores_key_values,len(scores_key_values))
#pRINT
fis = model.feature_importances_
print(fis,len(fis))
sorted_idx = fis.argsort()
print(fis[sorted_idx])
# print(sorted_idx)
# plt.barh([i for i in range(0,X_train.shape[1])][sorted_idx], model.feature_importances_[sorted_idx])
# plt.xlabel("Xgboost Feature Importance")
# print("Score key values",model2.get_booster().get_score(importance_type='gain'))
index_non_zero = list()
for i in range(len(scores_key_values.keys())):  # getting indices of used features in xgboost in [0, len(indices)]
    # converting scores_key_values.keys() to list(scores_key_values) and appending the indices of the features to index_non_zero without
    # the f character=
    index_non_zero.append(np.int64(list(scores_key_values)[i][1:]))  # indices of keys
sorted_values = np.argsort(scores_key_values.values())[::-1]  # argsorting based on gain and getting corresponding top indices.
print('PRINTING SORTED VALUES', sorted_values, 'END.', '\n')
print('INDEX NON ZERO', index_non_zero)

# from_top_temp = indices[np.array(index_non_zero)[sorted_values]]  # in range [0,125041]
# zir_from_top = np.array(list(set(indices) ^ set(indices[np.array(index_non_zero)[sorted_values]])))
# from_top = np.concatenate((from_top_temp, zir_from_top), axis=0)
# return from_top