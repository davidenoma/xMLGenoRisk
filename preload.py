import sys

import numpy as np
import pandas as pd


def main(genotype_file, phenotype_file):
    # The full genotype file
    # genotype_file_full = pd.read_csv(genotype_file, sep=" ", header=None)

    # The input file is a numpy file and will have indices from 0 to the number of snps
    # so the link with the indices_from_main_file will direct us to the  snps.
    df = pd.read_csv(genotype_file, sep=",", chunksize=10)
    # df = df.drop([df.shape[1] - 1], axis=1)
    y = list()
    for data in df:
        y.append(data)
    genotype_file_full2 = pd.concat([data for data in y], ignore_index=True)
    # so we write it to file for future reference
    input_indices_from_main_file = genotype_file_full2.columns.values
    input_indices_from_main_file = pd.DataFrame(input_indices_from_main_file)
    input_indices_from_main_file.to_csv('input_indices_from_main_file')
    # X = genotype_file_full2.values.astype(np.int64)


main(sys.argv[1], sys.argv[2])