import numpy as np
import pandas as pd
from numpy import array


def compute_mean(fi_scores):
    mean_score = np.mean(fi_scores)
    return mean_score



def drop_and_sort(fi_scores,mean_score):
    for i in range(fi_scores):
        if fi_scores[i] < mean_score:
            fi_scores.drop(labels=['i'])
            fi_scores.sort(descending=True)





#converting the indices to snps
def prepare_header_snps(snps_file_name):
    snps_file_name = 'snps_list_shorter.txt'
    header_file = pd.read_csv(snps_file_name, header=None)
    final_snps = []
    optimal_indices_2 = array([6, 14, 109, 160, 236, 285, 299, 300, 307, 318, 323, 339, 413,
                               429, 445, 450, 464, 475, 484, 513, 544, 564, 581, 584, 604, 613,
                               614, 646, 680, 687, 704, 724, 735, 743, 772, 784, 798, 826, 849,
                               856, 867, 874, 886, 893, 922, 940, 943, 983, 996, 997])
    header_file_subset = header_file.iloc[:, optimal_indices_2]

    for i in range(header_file_subset.shape[1]):
        final_snps.append(str(header_file_subset.iloc[0, i]).split('_')[0])
    header_file_string = str(header_file_subset)


#import scores from the different algorithms
XGBOOST_TOP_SNPS = pd.read_csv('xgboost_fp/top_1_percent.list')
RANDOM_FOREST_TOP_SNPS = pd.read_csv('randomforest/top_1_percent.list')
DL_TOP_SNPS = pd.read_csv('FeatureImportanceDL/top_1_percent.list')
DL_TOP_SNPS = pd.read_csv('deep_learning/top_1_percent.list')

xgboost_fi_scores = pd.read_csv('xgboost_fp/top_1_percent.list')
random_forest_fi_scores = pd.read_csv('randomforest/top_1_percent.list')
dl_fi_scores = pd.read_csv('FeatureImportanceDL/top_1_percent.list')


#compute the mean of feature importance vectors
fv_one = compute_mean(xgboost_fi_scores)
fv_two = compute_mean(random_forest_fi_scores)
fv_three = compute_mean(dl_fi_scores)

#Drop the features that are less that the mean of thesnps

drop_and_sort(xgboost_fi_scores,fv_one)
drop_and_sort(random_forest_fi_scores,fv_two)
drop_and_sort(dl_fi_scores,fv_three)

#Take top 1%

#Take top 5%

#Take top 10%



#find the set of overlap








#locate this overlap to snps



#use this snp set for risk prediction on the validation set




