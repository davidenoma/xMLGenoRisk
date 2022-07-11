# -*- coding: utf-8 -*-
"""HyperParameter_Tuning_OLOMO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11GHQ5dWk1bXPGPkqh6OLk2MskI_1eJ52
"""

# !pip install --quiet optuna
from array import array

import optuna
import optuna.integration
import optuna.visualization
import plotly

import xgboost as xgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

# from google.colab import drive
# drive.mount('/content/drive')

# Loading the dataset
from xgboost import XGBClassifier

X = pd.read_csv("../Xsubset.csv", header=None)
y = pd.read_csv("../hapmap_phenotype_recoded", header=None)
y.replace([1, 2], [0, 1], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=234)

"""## First  using the evaluation metric "Accuracy"""

def objective_accuracy(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # The booster parameter sets the type of learner. Usually this is either a tree
        # or a linear function. In the case of trees, the model will consist of an ensemble
        # of trees. For the linear booster, it will be a weighted sum of linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.6),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        # others are seed, random_state, n_jpbs
    }
    # Call Back for Pruning
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    bst = xgb.train(params, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback], verbose_eval=False)
    preds = bst.predict(dtest)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy

# optimizing for accuracy
study = optuna.create_study(direction="maximize")
study.optimize(objective_accuracy, n_trials=100, timeout=600, show_progress_bar=False)
trial_acc = study.best_trial
plot_param_imp = optuna.visualization.plot_param_importances(study)
# plot_param_imp.show()
plot_opt_history = optuna.visualization.plot_optimization_history(study)
# plot_opt_history.show()
print("Accuracy : %f", trial_acc.value)
print("Best HyperParamteters : %f", trial_acc.params)

"""
# XGBClassifier
## Using parameters from Accuracy
"""
d = xgb.DMatrix(X, label=y)
# trial.params
cv_results = xgb.cv(dtrain=d, params=trial_acc.params, nfold=5, num_boost_round=50, early_stopping_rounds=10,
                    metrics="auc",
                    as_pandas=True, seed=234)
print("CV RESULT", cv_results)

#The model feature importances without optimiziton
# imp_model = xgb.train(X, label=y)
# print(imp_model.feature_importances_)
imp_model = XGBClassifier(use_label_encoder=False)
imp_model.fit(X, y.values.ravel())
fi_scores = imp_model.feature_importances_
print(fi_scores)
scores_key_values = imp_model.get_booster().get_score(importance_type='gain')
print(scores_key_values)
plt.barh(list(scores_key_values.keys()), list(scores_key_values.values()))
plt.xlabel("Xgboost Feature Importance not optimized")
plt.show()


#The model feature importances with optimiztion
imp_model_opt = XGBClassifier(**trial_acc.params,use_label_encoder=False)
imp_model_opt.fit(X, y.values.ravel())
fi_scores_opt = imp_model_opt.feature_importances_
print(fi_scores_opt)
scores_opt_key_values = imp_model_opt.get_booster().get_score(importance_type='gain')
print(scores_opt_key_values)
print(imp_model_opt.feature_importances_)
plt.barh(list(scores_opt_key_values.keys()), list(scores_opt_key_values.values()))
plt.xlabel("Xgboost Feature Importance Optimized")
plt.show()
#sorting the features
sorted_idx = imp_model_opt.feature_importances_.argsort()
plt.barh(list(scores_opt_key_values.keys())[array(sorted_idx)], list(scores_opt_key_values.values())[array(sorted_idx)])

# optimizing for recall
study = optuna.create_study(direction="maximize")
study.optimize(objective_recall, n_trials=100, timeout=600, show_progress_bar=False)
trial = study.best_trial
print("Recall : %f", trial.value)
print("Best HyperParamteters : %f", trial.params)






"""## Using parameters from Recall"""


cv_results = xgb.cv(dtrain=d, params=trial.params, nfold=4, num_boost_round=50, early_stopping_rounds=10, metrics="auc",
                    as_pandas=True, seed=234)
cv_results
