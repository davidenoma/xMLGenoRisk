
# random forest for feature importance on a classification problem
import sys
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# define dataset
import load_dataset
X = sys.argv[1]
Y = sys.argv[2]
X,Y = load_dataset.main(X,Y)
X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.2)
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, Y)
# get importance
importance = model.feature_importances_
# summarize feature importance
print(importance)
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)

# pyplot.show()

# %%
# Data generation and model fitting
# ---------------------------------
# We generate a synthetic dataset with only 3 informative features. We will
# explicitly not shuffle the dataset to ensure that the informative features
# will correspond to the three first columns of X. In addition, we will split
# our dataset into training and testing subsets.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
# %%
# A random forest classifier will be fitted to compute the feature importances.
from sklearn.ensemble import RandomForestClassifier

feature_names = [f'feature {i}' for i in range(X_train.shape[1])]
feature_names = X_train.columns
forest = RandomForestClassifier(random_state=10)
forest.fit(X_train, y_train)

# %%
# Feature importance based on mean decrease in impurity
# -----------------------------------------------------
# Feature importances are provided by the fitted attribute
# `feature_importances_` and they are computed as the mean and standard
# deviation of accumulation of the impurity decrease within each tree.
#
# .. warning::
#     Impurity-based feature importances can be misleading for high cardinality
#     features (many unique values). See :ref:`permutation_importance` as
#     an alternative below.

import numpy as np
importances = forest.feature_importances_
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

from sklearn.inspection import permutation_importance
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