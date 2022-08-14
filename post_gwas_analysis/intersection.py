import pickle
import sys

import numpy as np
import pandas as pd
from sqlalchemy.dialects.mssql.information_schema import columns

f = open('best_indices_cvt_auc_recall3.pckl', 'rb')

best_indices = pickle.load(f)
f.close()
indices_new = []
temp = list()

for j in range(len(best_indices)):
    for k in range(len(best_indices[j])):
        #        temp.append(len(best_indices[j][k]))
        # print(best_indices[j][k])
        indices_new.append(list(best_indices[j][k]))
# print(indices_new)
indices_new1 = np.unique(np.concatenate(indices_new))


#reading in the the indices from indices genotype full 2
indices_from_gen_full2 = pd.read_csv('indices_from_gen_full2.csv',nrows=1,low_memory=False)
indices_from_gen_full2 = indices_from_gen_full2.columns.values[1:]
indices_from_gen_full2_list = list()
for i in range(indices_from_gen_full2.shape[0]):
    indices_from_gen_full2_list.append(int(indices_from_gen_full2[i]))

snps_list_file = pd.read_csv('snps_list_on_file.csv')
snps_from_gen_full2 = snps_list_file.iloc[indices_from_gen_full2_list,1]
#we now have the snps features from the genotype file.
# now we must index them according to numbers naturally as was in the input to the algorithm

print(snps_from_gen_full2.shape,indices_new1.shape)
snps_from_gen_full2.reset_index(drop=True,inplace=True)

#now we index the list according the best indices of the algorithm
final_snps_on_file = snps_from_gen_full2.iloc[indices_new1]
final_snps_on_file = final_snps_on_file.to_frame(name='rs')

#sum statistics from gemma associaion

sum_stats_gemma_bonf_p = pd.read_csv('sum_stats_gemma_bonf_p.csv')
sum_stats_gemma_bonf_p = sum_stats_gemma_bonf_p.loc[:,['rs','p_wald','chr']]

#sum statistics from logistic with covars
sum_stats_lr_p_values_bonf_p = pd.read_csv('sum_stats_lr_p_values_bonf_p.csv')
sum_stats_lr_p_values_bonf_p = sum_stats_lr_p_values_bonf_p.loc[:,['SNP','P', 'CHR']]
sum_stats_lr_p_values_bonf_p.columns = ['rs','p','chr']


#sum sttistics adjusted assoc
sum_stats_adj_ass_bonf_p = pd.read_csv('sum_stats_adj_ass_bonf_p.csv')
sum_stats_adj_ass_bonf_p = sum_stats_adj_ass_bonf_p.loc[:,['SNP','P', 'CHR']]
sum_stats_adj_ass_bonf_p.columns = ['rs','p','chr']

#merging with the raw data

intersect = pd.merge(final_snps_on_file,sum_stats_gemma_bonf_p,how='inner',on=['rs'])
intersect2 = pd.merge(final_snps_on_file,sum_stats_lr_p_values_bonf_p,how='inner',on=['rs'])
intersect3 = pd.merge(final_snps_on_file,sum_stats_adj_ass_bonf_p,how='inner',on=['rs'])


sum_stats_log = pd.read_csv('C:/Users/HP/OneDrive/Desktop/PROJECT RESULTS/results_log_sorted.csv')
sum_stats_log = sum_stats_log.loc[:,['SNP','P', 'CHR']]
sum_stats_log.columns = ['rs','p','chr']
# print(intersect)
# print(intersect2)
# print(intersect3)

intersect.to_csv('intersect_gemma.csv')
intersect2.to_csv('intersect_lr.csv')
intersect3.to_csv('intersect_ass_adj.csv')


# print(intersect)