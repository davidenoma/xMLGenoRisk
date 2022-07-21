import sys

import pandas as pd
import numpy as np


def main(X,Y):
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
    return X,Y
# if __name__ == '__main__':
#     main(sys.argv[1],sys.argv[2])