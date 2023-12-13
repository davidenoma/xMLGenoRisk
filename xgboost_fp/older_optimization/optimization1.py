#First we oprimize n_estimators = [50, 100], max_depth and learning_rate.
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# save numpy array as npz file
from numpy import asarray
from numpy import savez_compressed


# fixing seed: important to have same random train and test split as the optimizing
np.random.seed(0)

# loading data
#  creating random genotyped data with the same size as the data used in the original manuscript
# Note heterozygous and homozygous minor a
# re encoded as 0, 1, and 2, respectively.
#696, 125041
#X = np.random.randint(3, size=(200, 1000))
#Y = np.random.randint(2, size=(200, ))
def main(X,Y):
    #Load and convert to numpy
    # X = pd.read_csv(X, header=None)
    columns_to_skip = ['FID', 'IID', 'PAT', 'MAT', 'SEX','PHENOTYPE']
    df = pd.read_csv(X, chunksize=50, skiprows=1, header=None, low_memory=False, sep=" ", dtype='int8',
                     usecols=lambda column: column not in columns_to_skip, )
    y = list()
    counter = 1
    for data in df:
        # removing the extreme snp
        data = data.drop([data.shape[1] - 1], axis=1)
        print("Chunk Number: ", counter)
        y.append(data)
        counter = counter + 1
    final = pd.concat([data for data in y], ignore_index=True)
    X = final
    print(X, X.head(), X.shape)

    # save numpy array as npz file
    savez_compressed('genotype.npz', X)

    print(' DONE READING')
    Y = pd.read_csv(Y, header=None)
    Y.replace([1, 2], [0, 1], inplace=True)
    Y = Y.values.astype(np.int64)
    Y  = Y.ravel()


    print(Y.shape,Y.dtype)

    #Conversion of phenotype
    # phenotype_generated = LabelEncoder().fit_transform(phenotype_generated)
    # Tuning
    NUM_TRIALS = 10
    n_estimators = [150, 200]
    max_depth = [2, 4, 6, 8]
    learning_rate = [0.001, 0.01, 0.1]


    # For the parameter tuning.
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
    # # threads = os.sched_setaffinity(0) - 1
    # print(os.cpu_count())
    model = XGBClassifier(seed=0, nthread=8)
    tot_grid_results = list()
    best_grid_results = list()

    for i in range(NUM_TRIALS):
        print('testing trial',i)
        # Dividing into training and test set.x
        x, x_cv, y, y_cv = train_test_split(X, Y, test_size=0.2, train_size=0.8, stratify=Y,
                                            random_state=i)
        # optimizing xgboost parameters: never seen on x_cv and y_cv
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=16, cv=cv, verbose=1)
        grid_result = grid_search.fit(x, y)
        tot_grid_results.append(grid_result)
        best_grid_results.append([grid_result.best_score_, grid_result.best_params_])

    # save the hyperparamters
    f = open('tot_grid_results_stage1_kuopio_1.pckl', 'wb')
    pickle.dump(tot_grid_results, f)
    f.close()

    f = open('best_grid_results_stage1_kuopio_1.pckl', 'wb')
    pickle.dump(best_grid_results, f)
    f.close()

main(sys.argv[1],sys.argv[2])