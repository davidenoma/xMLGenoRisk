import os
import pandas as pd
from numpy import array
header_file = pd.read_csv('snps_list_shorter.txt',header=None)
final_snps = []
optimal_indices_2 = array([  6,  14, 109, 160, 236, 285, 299, 300, 307, 318, 323, 339, 413,
       429, 445, 450, 464, 475, 484, 513, 544, 564, 581, 584, 604, 613,
       614, 646, 680, 687, 704, 724, 735, 743, 772, 784, 798, 826, 849,
       856, 867, 874, 886, 893, 922, 940, 943, 983, 996, 997])
header_file_subset = header_file.iloc[:,optimal_indices_2]

for i in range(header_file_subset.shape[1]):
       print(i)
       final_snps.append(str(header_file_subset.iloc[0,i]).split('_')[0])
header_file_string = str(header_file_subset)
# print(header_file_subset, header_file_string)