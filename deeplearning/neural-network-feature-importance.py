
# Imports
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
# Load the data
# iris = load_iris()
# X = iris.data
# y = iris.target

genotype_generated = pd.read_csv('../Xsubset.csv', header=None)
phenotype_generated = pd.read_csv('../hapmap_phenotype_recoded',header=None)
genotype_generated = genotype_generated.to_numpy()
phenotype_generated = phenotype_generated.to_numpy()
phenotype_generated = phenotype_generated.reshape(len(phenotype_generated),)

# Create a train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train ,y_test = train_test_split(genotype_generated,phenotype_generated,test_size=0.2)
# X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=0.11)
# Preprocess the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Create a classifier
# clf = MLPClassifier(hidden_layer_sizes=(8, 4),learning_rate_init=0.01)
clf = MLPClassifier(hidden_layer_sizes=(150,100,50),learning_rate_init=0.01,max_iter=300,solver='adam',activation = 'relu',random_state=1)
# Fit the classifier using the training set
clf.fit(X_train, y_train)
# Evaluate the classifier using the test set
y_pred=clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
def get_feature_importance(j, n):
  s = accuracy_score(y_test, y_pred) # baseline score
  total = 0.0
  for i in range(n):
    perm = np.random.permutation(range(X_test.shape[0]))
    X_test_ = X_test.copy()
    X_test_[:, j] = X_test[perm, j]
    y_pred_ = clf.predict(X_test_)
    s_ij = accuracy_score(y_test, y_pred_)
    total += s_ij
  return s - total / n

# Feature importances
percent = 10
f = []
for j in range(X_test.shape[1]):
  f_j = get_feature_importance(j, 100)
  f.append(f_j)
# Plot

print(f[0:int(len(f)/percent)])
plt.figure(figsize=(10, 5))
plt.bar(range(int(X_test.shape[1]/percent)), f[0:int(len(f)/percent)], color="r", alpha=0.7)
plt.xticks(ticks=range(int(X_test.shape[1]/percent)))

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature importances (HAPMAP data set)")
plt.show()




