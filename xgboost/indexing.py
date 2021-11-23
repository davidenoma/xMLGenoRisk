

import pandas as pd
import  numpy as np
X = pd.read_csv('Xsubset.csv')
Y = pd.read_csv('hapmap_phenotype_recoded')

# print(X.columns)
#
# X=X.iloc[1:,]
# X=X[1:]
#
# print(X.index)
X.columns = np.arange(X.shape[1])
print(X)
print(np.int64(X))
print(X)
print(Y)
Y.shape
Y= np.int64(Y)

Y = Y.ravel()

print(Y)
# with open('Xsubset.csv') as f:
#     line = f.readline()
#     while line:
#         #
#         line = f.readline()
#         print(line[0])
#print(X)
#col = X.shape[0]
#row = X.shape[1]

#print(X.shape)
#nums = np.arange(0,row)
#prevIndex = X.columns
#X.columns = nums

#print(X.columns[1001])
