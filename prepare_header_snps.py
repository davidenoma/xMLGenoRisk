import os,sys,getopt
from os import write

import pandas as pd
from numpy import array

def prepare_header_snps(indices):
       final_snps = []
       header_file_subset = header_file.iloc[:, indices]
       for i in range(header_file_subset.shape[1]):
              final_snps.append(str(header_file_subset.iloc[0, i]).split('_')[0])
       header_file_string = str(header_file_subset)
       return header_file_string
       # print(header_file_subset, header_file_string)

def main(genotype_file,phenotype_file):
       # genotype_file = sys.argv[1]
       # phenotype_file = sys.argv[1]
       header_file = pd.read_csv(genotype_file)
       snps_list = list(header_file.columns.values)
       snps_list = prepare_header_snps(snps_list)
       with open('snps_list','w') as f:
              f.writelines(snps_list)
       f.close()



       genotype_file_full = pd.read_csv(genotype_file, header=None)
       phenotype_file = pd.read_csv(phenotype_file, header=None)
       print(phenotype_file)


main(sys.argv[1],sys.argv[2])