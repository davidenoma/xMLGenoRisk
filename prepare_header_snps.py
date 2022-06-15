import os,sys,getopt
from os import write

import pandas as pd
from numpy import array

#Remove the underscore from the snps
def rename_header_snps(snps_list):
       final_snps = []
       for i in range(0,len(snps_list)):
              final_snps.append(snps_list[i].split('_')[0])

       return  final_snps

def prepare_snps(indices):
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
       header_file = pd.read_csv(genotype_file,sep=" ")
       snps_list = list(header_file.columns.values)
       snps_list = rename_header_snps(snps_list)
       snps_list = pd.DataFrame(snps_list)
       #removing the extreme snp
       snps_list = snps_list.drop([snps_list.shape[0]-1], axis=0)
       # Writing to file
       # snps_list.to_csv('snps_list_on_file')
       # with open('snps_list','w') as f:
       #        f.writelines(snps_list)
       #        f.writelines(",")
       # f.close()

       #The full genotype file
       genotype_file_full = pd.read_csv(genotype_file, sep=" ", header=None)
       # removing the extreme snp
       genotype_file_full = genotype_file_full.drop([genotype_file_full.shape[0] - 1], axis=0)
       print(genotype_file_full,genotype_file_full.shape)


       #The phenotype file
       phenotype_file = pd.read_csv(phenotype_file, header=None)
       # print(phenotype_file)


main(sys.argv[1],sys.argv[2])