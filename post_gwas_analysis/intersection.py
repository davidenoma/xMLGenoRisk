import pickle

import numpy as np
import pandas as pd
f = open('best_indices_cvt_auc_recall3.pckl', 'rb')

best_indices = pickle.load(f)
f.close()
indices_new = []
temp = list()
print(best_indices[0])
for j in range(len(best_indices)):
    for k in range(len(best_indices[j])):
        #        temp.append(len(best_indices[j][k]))
        # print(best_indices[j][k])
        indices_new.append(list(best_indices[j][k]))
# print(indices_new)
indices_new1 = np.unique(np.concatenate(indices_new))


snps_list_file = pd.read_csv('snps_list_on_file.csv',index_col=None)
final_snps_on_file = snps_list_file.iloc[indices_new1,1]
# final_snps_on_file.to_csv('final_snps_on_file.csv')

#final snps on file using the indices from the other second module.py
final_snps_on_file = pd.read_csv('final_snps_on_file.csv')



#sum statistics from gemma associaion
sum_stats_filter_p_values = pd.read_csv('sum_stats_filter_p_values.csv')
sum_stats_filter_p_values_5_per = pd.read_csv('sum_stats_filter_p_values_0.05.csv')
snps_and_p_vals = sum_stats_filter_p_values.loc[:,['rs','p_wald']]
sum_stats_filter_p_values_5_per = sum_stats_filter_p_values_5_per.loc[:,['rs','p_wald','chr']]



final_snps_on_file.columns = ['rs']


intersect = pd.merge(final_snps_on_file,snps_and_p_vals,how='inner',on=['rs'])
intersect2 = pd.merge(final_snps_on_file,sum_stats_filter_p_values_5_per,how='inner',on=['rs'])
intersect2.to_csv('intersect_xg_gemma_0.05.csv')
print(intersect)