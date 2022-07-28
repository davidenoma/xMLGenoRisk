import pandas as pd


def calc_and_save_feature_imp_scores(f,X_test):

  top_features_idx = dict(zip([x for x in range(X_test.shape[1])], f))
  top_features_sorted_idx = pd.DataFrame.from_dict(top_features_idx, orient='index')
  top_features_sorted_idx.columns = ['Importance_scores']
  top_features_sorted_idx = top_features_sorted_idx.sort_values(by=["Importance_scores"], ascending=False)
  top_length = top_features_sorted_idx.shape[0]
  top_1_percent = top_features_sorted_idx.iloc[:int(top_length / 100), :]
  top_5_percent = top_features_sorted_idx.iloc[:int(top_length / 20), :]
  top_50 = top_features_sorted_idx.iloc[:50, :]
  top_1_percent.to_csv('deeplearning/top_1_percent.list')
  top_50.to_csv('top_100.list')

  return top_50,top_1_percent,top_5_percent
