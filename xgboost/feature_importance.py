def cal_XGboost_feature_importance(X_train, Y_train, indices, model, X_test, Y_test):
    # # initial sort: algorithm 1 step 1
    model_1 = clone(model)
    eval_set = [(X_test[:, indices], Y_test)]
    model_1.fit(X_train[:, indices], Y_train, verbose=False, early_stopping_rounds=(model_1.n_estimators) / 10,
                eval_metric="auc", eval_set=eval_set)
    # now sort based on gain
    scores_key_values = model_1.get_booster().get_score(importance_type='gain')
    index_non_zero = list()
    for i in range(len(scores_key_values.keys())):  # getting indices of used features in xgboost in [0, len(indices)]
        # converting scores_key_values.keys() to list(scores_key_values) and appending the indices of the features to index_non_zero without
        # the f character=
        index_non_zero.append(np.int64(list(scores_key_values)[i][1:]))  # indices of keys
    sorted_values = np.argsort(scores_key_values.values())[
                    ::-1]  # argsorting based on gain and getting corresponding top indices.
    print('PRINTING SORTED VALUES', sorted_values, 'END.', '\n')
    print('INDEX NON ZERO', index_non_zero)

    from_top_temp = indices[np.array(index_non_zero)[sorted_values]]  # in range [0,125041]
    zir_from_top = np.array(list(set(indices) ^ set(indices[np.array(index_non_zero)[sorted_values]])))
    from_top = np.concatenate((from_top_temp, zir_from_top), axis=0)
    return from_top