import os, sys, getopt
from os import write
import numpy as np
import pandas as pd
from numpy import array
from pyspark.sql import SparkSession


def rename_header_snps(snps_list):
    final_snps = []
    for i in range(0, len(snps_list)):
        final_snps.append(snps_list[i].split('_')[0])

    return final_snps


def prepare_snps(indices):
    final_snps = []
    header_file_subset = header_file.iloc[:, indices]
    for i in range(header_file_subset.shape[1]):
        final_snps.append(str(header_file_subset.iloc[0, i]).split('_')[0])
    header_file_string = str(header_file_subset)
    return header_file_string
    # print(header_file_subset, header_file_string)


def generating_snps_list(genotype_file):
    # Generating the snps list

    header_file = pd.read_csv(genotype_file, sep=" ", nrows=2)
    snps_list = list(header_file.columns.values)
    snps_list = rename_header_snps(snps_list)
    snps_list = pd.DataFrame(snps_list)

    # removing the extreme snp

    snps_list = snps_list.drop([snps_list.shape[0] - 1], axis=0)
    # Writing to file
    snps_list.to_csv('snps_list_on_file.csv')
    print('Done writing snps')


def loading_with_pyspark(genotype_file):
    # using pyspark because of the memory consumption
    spark = SparkSession.builder.config('spark.sql.debug.maxToStringFields', 2000).config("spark.driver.memory",
                                                                                          "120g").getOrCreate()
    pdf = spark.read.options(maxColumns=2000000).csv(genotype_file, sep=" ", header=None, nullValue='NA')
    genotype_file_full = pdf.toPandas()
    genotype_file_full.columns = [i for i in range(genotype_file_full.shape[1])]



def main(genotype_file, phenotype_file):
    # The full genotype file
    # genotype_file_full = pd.read_csv(genotype_file, sep=" ", header=None)

    # applying chunking with pandas because of memory.
    # It would be useful to compare the memory consumption differences also with pyspark
    df = pd.read_csv("42snps", sep=" ", chunksize=10, header=None)
    y = list()
    for data in df:
        # removing the extreme snp
        data = data.drop([data.shape[1] - 1], axis=1)
        y.append(data)
    final = pd.concat([data for data in y], ignore_index=True)
    # remove the column with the snps name since we already have it on file.
    genotype_file_full = final.drop([0], axis=0)

    genotype_file_full = final.drop([0], axis=0)
    # Removing snps that are missing individuals
    genotype_file_full = genotype_file_full.dropna(axis=1)
    genotype_file_full.to_csv('genotype_file_full2', index=False)
    # The phenotype file
    # phenotype_file = pd.read_csv(phenotype_file, header=None)

    # The input file is a numpy file and will have indices from 0 to the number of snps
    # so the link with the indices_from_main_file will direct us to the  snps.

    #so we write it to file for future reference
    input_indices_from_main_file = genotype_file_full2.columns.values
    input_indices_from_main_file = pd.DataFrame(input_indices_from_main_file)
    input_indices_from_main_file.to_csv('input_indices_from_main_file')

    #then to optimization
    #X = genotype_file_full2.values.astype(np.int64)
main("42snps", "hapmap_phenotype_recoded")
# main(sys.argv[1],sys.argv[2])
