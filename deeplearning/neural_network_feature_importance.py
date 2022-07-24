
# Imports
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

X = sys.argv[1]
Y = sys.argv[2]
print(X,Y)
df = pd.read_csv(X, chunksize=5, header=None, low_memory=False, verbose=True)
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
X = X.values.astype(np.int64)
# we need the values without the numpy header
X = X[1:, :]

# save numpy array as npz file
# savez_compressed('genotype.npz', X)
print(' DONE READING')

Y = pd.read_csv(Y, header=None)
Y.replace([1, 2], [0, 1], inplace=True)
Y = Y.values.astype(np.int64)
Y = Y.ravel()
print(Y.shape, Y.dtype)



# Create a train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.2)
# X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=0.11)
# Create a classifier
# clf = MLPClassifier(hidden_layer_sizes=(8, 4),learning_rate_init=0.01)
clf = MLPClassifier(hidden_layer_sizes=(150,100,50),learning_rate_init=0.01,max_iter=300,solver='adam',activation = 'relu',random_state=1)
# Fit the classifier using the training set
clf.fit(X_train, y_train)
# Evaluate the classifier using the test set

def get_feature_importance(j, n):
  y_pred = clf.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  print(acc)
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
#Permutation feature importance scores
print(f[0:int(len(f)/percent)])
plt.figure(figsize=(10, 5))


plt.bar(range(int(X_test.shape[1]/percent)), f[0:int(len(f)/percent)], color="r", alpha=0.7)
plt.xticks(ticks=range(int(X_test.shape[1]/percent)))

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature importances (GPRCA data set)")
plt.show()
plt.savefig("Feature importances (GPRCA data set)")




