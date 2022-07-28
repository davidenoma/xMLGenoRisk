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

# from fast_ml.model_development import train_valid_test_split
#
# X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'SalePrice',
#                                                                             method='sorted', sort_by_col='saledate',
#                                                                             train_size=0.8, valid_size=0.1, test_size=0.1)

# Create a train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=0.11)


# Create a classifier
# clf = MLPClassifier(hidden_layer_sizes=(8, 4),learning_rate_init=0.01)
clf = MLPClassifier(hidden_layer_sizes=(300, 150, 50, 1), max_iter=500, learning_rate_init=0.01, solver='adam',
                    activation='logistic', verbose=True)
# Fit the classifier using the training set
clf.fit(X_train, y_train)


# Evaluate the classifier using the test set


def get_feature_importance(j, n):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    s = accuracy_score(y_test, y_pred)  # baseline score
    total = 0.0
    for i in range(n):
        perm = np.random.permutation(range(X_test.shape[0]))
        X_test_ = X_test.copy()
        X_test_[:, j] = X_test[perm, j]
        y_pred_ = clf.predict(X_test_)
        s_ij = accuracy_score(y_test, y_pred_)
        total += s_ij
    return s - total / n


f = []
for j in range(X_test.shape[1]):
    f_j = get_feature_importance(j, 100)
    f.append(f_j)


# Permutation feature importance scores
def calc_and_save_feature_imp_scores(f, X_test):
    top_features_idx = dict(zip([x for x in range(X_test.shape[1])], f))
    top_features_sorted_idx = pd.DataFrame.from_dict(top_features_idx, orient='index')
    top_features_sorted_idx.columns = ['Importance_scores']
    top_features_sorted_idx = top_features_sorted_idx.sort_values(by=["Importance_scores"], ascending=False)
    top_length = top_features_sorted_idx.shape[0]
    top_1_percent = top_features_sorted_idx.iloc[:int(top_length / 100), :]
    top_5_percent = top_features_sorted_idx.iloc[:int(top_length / 20), :]
    top_50 = top_features_sorted_idx.iloc[:50, :]

    return top_50, top_1_percent, top_5_percent

top_50, top_1_percent = calc_and_save_feature_imp_scores(f, X_test)

top_1_percent.to_csv('deeplearning/top_1_percent.list')
top_50.to_csv('top_100.list')

# Plot
plt.figure(figsize=(10, 5))
plt.bar([str(x) for x in list(top_50.index)], top_50['Importance_scores'], color="r", alpha=0.7)
# plt.xticks(ticks=list(top_5_percent.index))
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.title("Feature importances (GPRCA data set)")
plt.savefig("Feature importances (GPRCA data set).png", dpi=300)
plt.show()
