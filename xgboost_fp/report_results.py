import sys
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# fixing seed: important to have same random train and test split as the optimizing
np.random.seed(0)

def all_results_SVM(XX_train, YY_train, XX_validation, YY_validation, indices):
    classifier = svm.SVC(probability=True, random_state=3, kernel='linear', C=1.5, class_weight='balanced')
    classifier.fit(XX_train[:, indices], YY_train)
    ts_score = classifier.predict_proba(XX_validation[:, indices])
    # print(ts_score)
    return ts_score[:, 1]


def main(X,Y):
    # Load and convert to numpy
    # X = pd.read_csv(X, header=None)
    df = pd.read_csv(X, chunksize=5, header=None, low_memory=False, verbose=True)
    y = list()
    counter = 1
    for data in df:
        # removing the extreme snp
        data = data.drop([data.shape[1] - 1], axis=1)
        print("Chunk Number: ", counter)
        y.append(data)
        counter = counter + 1
    final = pd.concat([data for data in y], ignore_index=True)
    X = final
    X = X.values.astype(np.int64)
    # we need the values without the numpy header
    X = X[1:, :]

    # save numpy array as npz file
    # savez_compressed('genotype.npz', X)

    print(' DONE READING')
    Y = pd.read_csv(Y, header=None)
    Y.replace([1, 2], [0, 1], inplace=True)
    Y = Y.values.astype(np.int64)
    Y = Y.ravel()
    print(Y.shape, Y.dtype)
    print(Y.shape, Y.dtype)

    # loading best indices from the output of the clean_second_module.py
    f = open('best_indices_cvt_auc_recall3.pckl', 'rb')
    best_indices = pickle.load(f)

    f.close()

    indices_new = []
    temp = list()
    for j in range(len(best_indices)):
        for k in range(len(best_indices[j])):
            #        temp.append(len(best_indices[j][k]))
            indices_new.append(list(best_indices[j][k]))
    # print(indices_new)
    indices_new1 = np.unique(np.concatenate(indices_new))
    print(indices_new1)

    NUM_TRIALS = 10
    counter = -1

    #Mean Average Precision

    tot_average_precisionTR = list()
    tot_average_precisionDev = list()
    tot_average_precisionTS = list()

    #average AUC
    tot_average_AUCTR = list()
    tot_average_AUCDev = list()
    tot_average_AUCTS = list()


    indices_ID = range(X.shape[0])
    for i in range(NUM_TRIALS):
        print(i)
        x, x_cv, y, y_cv, indices_x, indices_x_cv = train_test_split(X, Y, indices_ID, test_size=0.2,train_size=0.8, stratify=Y, random_state=i)
        # optimizing xgboost parameters: never seen on x_cv and y_cv
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        svm_fpr_list = list()
        svm_tpr_list = list()
        __list = list()
        for train, test in cv.split(x, y):  # accessing train and test for stage 2 parameter tuning
            X_train = x[train]
            Y_train = y[train]
            X_test = x[test]
            Y_test = y[test]
            counter = counter + 1

            if (len(indices_new[counter])):
                # training
                ts_scoreL1 = all_results_SVM(X_train, Y_train, X_train, Y_train, indices_new[counter])
                tot_average_precisionTR.append(average_precision_score(Y_train, ts_scoreL1))
                tot_average_AUCTR.append(roc_auc_score(Y_train,ts_scoreL1))
                # val
                ts_scoreL1 = all_results_SVM(X_train, Y_train, X_test, Y_test, indices_new[counter])
                tot_average_precisionDev.append(average_precision_score(Y_test, ts_scoreL1))
                tot_average_AUCDev.append(roc_auc_score(Y_test, ts_scoreL1))
                # test
                ts_scoreL1 = all_results_SVM(X_train, Y_train, x_cv, y_cv, indices_new[counter])
                tot_average_precisionTS.append(average_precision_score(y_cv, ts_scoreL1))
                tot_average_AUCTS.append(roc_auc_score(y_cv, ts_scoreL1))




                # tot_average_ROC_TS.append(roc_(y_cv, ts_scoreL1))
                precision, recall, thresholds = precision_recall_curve(y_cv,ts_scoreL1)

                svm_auc = roc_auc_score(y_cv, ts_scoreL1)
                # print('SVM: AUC=%.3f' % (svm_auc))
                svm_fpr, svm_tpr, _ = roc_curve(y_cv, ts_scoreL1)

            print("len svm_fpr",len(svm_fpr))
            svm_fpr_list.append(svm_fpr)
            svm_tpr_list.append(svm_tpr)
            __list.append(_)

                # pyplot.plot(svm_fpr, svm_tpr,marker='.')
                # pyplot.savefig('')
                # pyplot.show()

    # print(len(svm_fpr_list),len(svm_fpr_list[0]),svm_fpr_list[0][0].shape)

    # print(len(tot_average_ROC_TS),len(tot_average_ROC_TS[0]),tot_average_ROC_TS[0][0].shape)


    # for i in range(len(tot_average_ROC_TS)):
    #         # print(i,len(tot_average_ROC_TS[i]),len(tot_average_ROC_TS[i][0]))
    #         svm_fpr_list.append(tot_average_ROC_TS[i][0])
    #         # print(svm_fpr_list,len(svm_fpr_list))
    #         svm_tpr_list.append(tot_average_ROC_TS[i][1])
    #         __list.append(tot_average_ROC_TS[i][2])
    # svm_fpr_list = np.array(svm_fpr_list)
    # for i in range(svm_fpr_list.shape[0]):
    #     print(len(svm_fpr_list[i]))
    #
    # print(svm_fpr_list.shape, np.mean(svm_fpr_list,axis=1))
    # print(len(svm_tpr_list), len(svm_tpr_list[50]))
    # # print(len(__list), len(__list[2]))
    #
    # svm_fpr_list,svm_tpr_list,__list
    # svm_fpr_mean = np.mean(svm_fpr_list)
    # svm_tpr_mean= np.mean(svm_tpr_list)
    # __mean = np.mean(__list)
    #
    # pyplot.plot(svm_fpr_mean, svm_tpr_mean, marker='.')
    # pyplot.show()



    # print((tot_average_ROC_TS)[1], len((tot_average_ROC_TS)[1]))
    print(str('Train Average precision: ') + str(np.mean(tot_average_precisionTR) * 100) + str('std: ') + str(
        np.std(tot_average_precisionTR)))
    print(str('Val Average precision: ') + str(np.mean(tot_average_precisionDev) * 100) + str('std: ') + str(
        np.std(tot_average_precisionDev)))
    print(str('Test Average precision: ') + str(np.mean(tot_average_precisionTS) * 100) + str('std: ') + str(
        np.std(tot_average_precisionTS)))

    print(np.mean(tot_average_AUCTR),np.mean(tot_average_AUCDev),np.mean(tot_average_AUCTS))
    #
    # print(str('Train Average AUC: ') + str(np.mean(tot_average_precisionTR) * 100) + str('std: ') + str(
    #     np.std(tot_average_precisionTR)))
    # print(str('Val Average precision: ') + str(np.mean(tot_average_precisionDev) * 100) + str('std: ') + str(
    #     np.std(tot_average_precisionDev)))
    # print(str('Test Average precision: ') + str(np.mean(tot_average_precisionTS) * 100) + str('std: ') + str(
    #     np.std(tot_average_precisionTS)))
    #



main(sys.argv[1],sys.argv[2])

