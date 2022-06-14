from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
# !pip install --quiet optuna
import optuna
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

X = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/Xsubset.csv", header=None)
y = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/hapmap_phenotype_recoded.txt", sep=" ", header=None)
y.replace([1, 2], [0, 1], inplace=True)

y


def objective(trial):
    X = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/Xsubset.csv", header=None)
    y = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/hapmap_phenotype_recoded.txt", sep=" ",
                    header=None)
    y.replace([1, 2], [0, 1], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=234)
    n_estimators = trial.suggest_int('n_estimators', 2, 20)
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # CALL BACK
    bst = clf.fit(X_train, y_train.values.ravel())
    preds = bst.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)
trial = study.best_trial

print("Accuracy : %f", trial.value)
print("Best HyperParamteters : %f", trial.params)

"""## Accuracy
0.7777777777777778

## Best HyperParamteters
'n_estimators': 5, 

'max_depth': 8.228592473942985

### RECALL
"""


def objective(trial):
    X = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/Xsubset.csv", header=None)
    y = pd.read_csv("/content/drive/My Drive/David_Enoma_PhD_Project/hapmap_phenotype_recoded.txt", sep=" ",
                    header=None)
    y.replace([1, 2], [0, 1], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=234)
    n_estimators = trial.suggest_int('n_estimators', 2, 20)
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # CALL BACK
    bst = clf.fit(X_train, y_train.values.ravel())
    preds = bst.predict(X_test)
    pred_labels = np.rint(preds)
    Recall = recall_score(y_test, pred_labels)
    return Recall


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)
trial = study.best_trial

print("Recall : %f", trial.value)
print("Best HyperParamteters : %f", trial.params)

"""## Recall 
1.0

## Best HyperParamteters 
'n_estimators': 18, 

'max_depth': 1.3768623504453061

Looks like using recall lead to overfitting


"""# RandomForestClassifier
## Using parameters from Accuracy
"""

clf = RandomForestClassifier(n_estimators=5, max_depth=8.228592473942985)

cvs = cross_val_score(clf, X, y, scoring='roc_auc', n_jobs=-1, cv=4).mean()
cvs

"""## Using parameters from Recall"""

clf = RandomForestClassifier(n_estimators=18, max_depth=1.3768623504453061)

cvs = cross_val_score(clf, X, y, scoring='roc_auc', n_jobs=-1, cv=4).mean()
cvs