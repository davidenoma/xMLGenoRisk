# After finding the optimal n_estimators, max_depth and learning_rate, now subsampling rate is optimized.
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import sys
# fixing seed: important to have same random train and test split as the optimizing
np.random.seed(0)

# loading data
#  creating random genotyped data with the same size as the data used in the original manuscript
# Note heterozygous and homozygous minor are encoded as 0, 1, and 2, respectively.
# Functions
def model_XGboost(n_estimators, max_depth, learning_rate):
    model_x = XGBClassifier(nthread=8, seed=0, n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate)
    return model_x

def main(X,Y):

    # X = pd.read_csv(X, header=None)
    df = pd.read_csv(X, chunksize=5, header=None, low_memory=False,verbose=True)
    y = list()
    counter = 1
    for data in df:
        # removing the extreme snp
        data = data.drop([data.shape[1] - 1], axis=1)
        print("Chunk Number: ",counter)
        y.append(data)
        counter = counter+1
    final = pd.concat([data for data in y], ignore_index=True)
    X=final
    X = X.values.astype(np.int64)
    #we need the values without the numpy header
    X = X[1:,:]
    print(' DONE READING')
    Y = pd.read_csv(Y, header=None)
    Y.replace([1, 2], [0, 1], inplace=True)
    Y = Y.values.astype(np.int64)
    Y  = Y.ravel()

    print(X,X.shape,X.dtype)
    print(Y.shape,Y.dtype)

    f = open('best_grid_results_stage1_kuopio_0.pckl', 'rb')
    best0 = pickle.load(f)
    f.close()

    f = open('best_grid_results_stage1_kuopio_1.pckl', 'rb')
    best1 = pickle.load(f)
    f.close()

    f = open('best_grid_results_stage1_kuopio_2.pckl', 'rb')
    best2 = pickle.load(f)
    f.close()

    # Tuning
    NUM_TRIALS = 10
    subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    param_grid = dict(subsample=subsample)

    # model = XGBClassifier(seed=0,nthread=1)
    tot_grid_results = list()
    best_grid_results = list()
    for i in range(NUM_TRIALS):
        print('testing trial',i)
        x, x_cv, y, y_cv = train_test_split(X, Y, test_size=0.2, train_size=0.8, stratify=Y,
                                            random_state=i)
        # optimizing xgboost parameters: never seen on x_cv and y_cv
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        # Comparison
        temp_vec_log = [-best0[i][0], -best1[i][0], -best2[i][0]]  # lower the better
        temp_n_est = [best0[i][1]['n_estimators'], best1[i][1]['n_estimators'], best2[i][1]['n_estimators']]
        temp_max_depth = [best0[i][1]['max_depth'], best1[i][1]['max_depth'], best2[i][1]['max_depth']]
        temp_lr = [best0[i][1]['learning_rate'], best1[i][1]['learning_rate'], best2[i][1]['learning_rate']]
        inx_best = np.argsort(temp_vec_log)[0]

        model = model_XGboost(temp_n_est[inx_best], temp_max_depth[inx_best], temp_lr[inx_best])

        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=16, cv=cv, verbose=1)
        grid_result = grid_search.fit(x, y)
        tot_grid_results.append(grid_result)
        best_grid_results.append([grid_result.best_score_, grid_result.best_params_])

    # save the hyperparamters
    f = open('tot_grid_results_stage1_kuopio_7.pckl', 'wb')
    pickle.dump(tot_grid_results, f)
    f.close()

    f = open('best_grid_results_stage1_kuopio_7.pckl', 'wb')
    pickle.dump(best_grid_results, f)
    f.close()

main(sys.argv[1],sys.argv[2])