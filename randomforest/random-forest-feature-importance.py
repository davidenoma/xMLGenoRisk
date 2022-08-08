# random forest for feature importance on a classification problem
# define dataset
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from loading_and_cleaning import load_dataset
from deeplearning.neural_network_feature_importance import calc_and_save_feature_imp_scores
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

X = sys.argv[1]
Y = sys.argv[2]

X, Y = load_dataset.load_data(X, Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, Y)
# get importance
importance = model.feature_importances_

# summarize feature importance
top_50, top_1_percent, top_5_percent = calc_and_save_feature_imp_scores(importance, X_test)
print(top_50, top_1_percent, top_5_percent, sep="\n")
top_50_num = top_50.to_numpy()
importance = top_50

top_1_percent.to_csv('randomforest/top_1_percent.list')
top_50.to_csv('randomforest/top_50.list')

# Plot
plt.figure(figsize=(10, 5))
plt.bar([str(x) for x in list(top_50_num.index)], top_50['Importance_scores'], color="r", alpha=0.7)
# plt.xticks(ticks=list(top_5_percent.index))
plt.xlabel("Feature")
plt.ylabel("Importance")
pyplot.show()


feature_names = [f'feature {i}' for i in range(X_train.shape[1])]
feature_names = X_train.columns
forest = RandomForestClassifier(random_state=10)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
importances = importances[1:50]
std = np.std([
    tree.feature_importances_ for tree in forest.estimators_], axis=0)

# %%
# Let's plot the impurity-based importance.
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# Feature importance based on feature permutation
# -----------------------------------------------
# Permutation feature importance overcomes limitations of the impurity-based
# feature importance: they do not have a bias toward high-cardinality features
# and can be computed on a left-out test set.
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=feature_names)

# %%
# The computation for full permutation importance is more costly. Features are
# shuffled n times and the model refitted to estimate the importance of it.
# Please see :ref:`permutation_importance` for more details. We can now plot
# the importance ranking.

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
