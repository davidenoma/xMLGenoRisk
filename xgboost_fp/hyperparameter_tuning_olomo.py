# -*- coding: utf-8 -*-
"""HyperParameter_Tuning_OLOMO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11GHQ5dWk1bXPGPkqh6OLk2MskI_1eJ52
"""

# !pip install --quiet optuna

import optuna
import optuna.integration

import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


# from google.colab import drive
# drive.mount('/content/drive')
"""## First  using the evaluation metric "Accuracy"""

def objective(trial):
  X = pd.read_csv("../Xsubset.csv",header=None)
  y = pd.read_csv("../hapmap_phenotype_recoded", header = None)
  y.replace([1,2], [0,1], inplace = True)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 234)
  dtrain = xgb.DMatrix(X_train, label = y_train)
  dtest = xgb.DMatrix(X_test, label = y_test)


  param = {
      "silent": 1,
      "objective": "binary:logistic",
      "eval_metric": "auc",
      # The booster parameter sets the type of learner. Usually this is either a tree 
      # or a linear function. In the case of trees, the model will consist of an ensemble 
      # of trees. For the linear booster, it will be a weighted sum of linear functions.
      "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
      "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
      "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
      
  }
  n_estimators = [50, 100]

  if param["booster"] == "gbtree" or param["booster"] == "dart":
    param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
    param["eta"]= trial.suggest_loguniform("eta", 1e-8, 1.0)
    param["gamma"]= trial.suggest_loguniform("gamma", 1e-8, 1.0)
    param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

  if param["booster"] == "dart":
    param["sample_type"]=trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    param["normalize_type"]=trial.suggest_categorical("normalize_type", ["tree", "forest"])
    param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
    param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)


    # Call Back for Pruning

  pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
  bst = xgb.train(param, dtrain, evals = [(dtest, "validation")], callbacks = [pruning_callback])
  preds = bst.predict(dtest)
  pred_labels = np.rint(preds)
  accuracy = accuracy_score(y_test, pred_labels)
  return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials = 100, timeout=600)
trial = study.best_trial

print("Accuracy : %f", trial.value)
print("Best HyperParamteters : %f", trial.params)

"""### Accuracy 
0.6111111111111112

### Best HyperParamteters 
'booster': 'gblinear', 

'lambda': 0.030475097247617086, 

'alpha': 3.753658128510404e-05

## For a problem as sensitive as Cancer, Evaluation metrics such as "Recall" or "F1 Score" are more important than "Accuracy" because the focus is on ensuring the highest number of people who have cancer are predicted to have cancer and so I will be using "Recall" as my evaluation metric here
"""
def objective(trial):
  X = pd.read_csv("../Xsubset.csv",header=None)
  y = pd.read_csv("../hapmap_phenotype_recoded", sep = " ", header = None)
  y.replace([1,2], [0,1], inplace = True)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 234)
  dtrain = xgb.DMatrix(X_train, label = y_train)
  dtest = xgb.DMatrix(X_test, label = y_test)
  param = {
      "verbosity": 0,
      # logistic regression for binary classification, output probability
      "objective": "binary:logistic",
      # use exact for small dataset.
      "tree_method": "auto",
      "eval_metric": "auc",
      # The booster parameter sets the type of learner. Usually this is either a tree 
      # or a linear function. In the case of trees, the model will consist of an ensemble 
      # of trees. For the linear booster, it will be a weighted sum of linear functions.
      # defines booster, gblinear for linear functions.
      "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
      # L2 regularization weight.
      "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log = True), 
      # L1 regularization weight.
      "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log = True),
      # sampling ratio for training data.
      "subsample": trial.suggest_float("subsample", 0.2, 1.0),
      # sampling according to each tree.
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
  }

  if param["booster"] == "gbtree" or param["booster"] == "dart":
    # maximum depth of the tree, signifies complexity of the tree.
    param["max_depth"] = trial.suggest_int("max_depth", 1, 7, step=2)
    # minimum child weight, larger the term more conservative the tree.
    param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
    # "eta" shrinks the feature weights to make the boosting process more conservative.
    param["eta"]= trial.suggest_float("eta", 1e-8, 1.0, log = True)
    # Minimum loss reduction required to make a further partition on a leaf node of the tree. 
    # The larger gamma is, the more conservative the algorithm will be.
    param["gamma"]= trial.suggest_float("gamma", 1e-8, 1.0, log = True)
    #  Controls the way new nodes are added to the tree.
    param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    
  if param["booster"] == "dart":
    # Type of sampling algorithm.
    param["sample_type"]=trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    # Type of normalization algorithm.
    param["normalize_type"]=trial.suggest_categorical("normalize_type", ["tree", "forest"])
    # Dropout rate (a fraction of previous trees to drop during the dropout).
    param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log = True)
    # Probability of skipping the dropout procedure during a boosting iteration.
    param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log = True)


  # Call Back for Pruning
  # Pruning is used to terminate unpromising trials early, so that computing time can be used for trials that show more potential.
  pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
  bst = xgb.train(param, dtrain, evals = [(dtest, "validation")], callbacks = [pruning_callback])
  preds = bst.predict(dtest)
  # np.rint means return integer. It rounds up elements to the nearest integer. 
  # This is done so we can use accuracy_score because "binary:logistic" outputs probability not discrete numbers [0,1] but the probability of each sample being 1
  pred_labels = np.rint(preds)
  Recall = recall_score(y_test, pred_labels)
  return Recall

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials = 100, timeout=600)
trial = study.best_trial

print("Recall : %f", trial.value)
print("Best HyperParamteters : %f", trial.params)

"""### Recall
1.0

### Best HyperParamteters
'booster': 'dart', 

'lambda': 8.669790267915281e-07, 

'alpha': 1.1536382592082632e-08, 

'subsample': 0.28469669085250404, 

'colsample_bytree': 0.5566590888822147, 

'max_depth': 7, 

'min_child_weight': 10, 

'eta': 0.003933096882701822, 

'gamma': 0.0018015367008443472, 

'grow_policy': 'lossguide', 

'sample_type': 'weighted', 

'normalize_type': 'forest', 

'rate_drop': 8.829825957910605e-06, 

'skip_drop': 5.189332052983798e-07}
"""

"""
# XGBRegressor
## Using parameters from Accuracy
"""

X = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/Xsubset.csv", header=None)
y = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/hapmap_phenotype_recoded.txt", sep=" ", header=None)
y.replace([1, 2], [0, 1], inplace=True)
d = xgb.DMatrix(X, label=y)

params = {'booster': 'gblinear', "objective": "binary:logistic", 'max_depth': 5, 'lambda': 0.030475097247617086,
          'alpha': 753658128510404e-05}

cv_results = xgb.cv(dtrain=d, params=params, nfold=4, num_boost_round=50, early_stopping_rounds=10, metrics="auc",
                    as_pandas=True, seed=234)
cv_results

"""## Using parameters from Recall"""

params = {'booster': 'dart', 'lambda': 8.669790267915281e-07, 'alpha': 1.1536382592082632e-08,
          'subsample': 0.28469669085250404, 'colsample_bytree': 0.5566590888822147,
          'max_depth': 7, 'min_child_weight': 10, 'eta': 0.003933096882701822, 'gamma': 0.0018015367008443472,
          'grow_policy': 'lossguide', 'sample_type': 'weighted',
          'normalize_type': 'forest', 'rate_drop': 8.829825957910605e-06, 'skip_drop': 5.189332052983798e-07}

cv_results = xgb.cv(dtrain=d, params=params, nfold=4, num_boost_round=50, early_stopping_rounds=10, metrics="auc",
                    as_pandas=True, seed=234)
cv_results