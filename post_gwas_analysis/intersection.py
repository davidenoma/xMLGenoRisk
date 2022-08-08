import pandas as pd

final_snps_on_file = pd.read_csv('final_snps_on_file.csv')
sum_stats_filter_p_values = pd.read_csv('sum_stats_filter_p_values.csv')
sum_stats_filter_p_values_5_per = pd.read_csv('sum_stats_filter_p_values_0.05.csv')

final_snps_on_file.columns = ['num','rs']
snps_and_p_vals = sum_stats_filter_p_values.loc[:,['rs','p_wald']]
sum_stats_filter_p_values_5_per = sum_stats_filter_p_values.loc[:,['rs','p_wald']]


intersect = pd.merge(final_snps_on_file,snps_and_p_vals,how='inner',on=['rs'])
intersect2 = pd.merge(final_snps_on_file,sum_stats_filter_p_values_5_per,how='inner',on=['rs'])
print(intersect,intersect2)