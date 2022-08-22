import inspect
import os
import sys

import numpy
from numpy import interp
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, recall_score, auc
from sklearn.metrics import roc_curve
from matplotlib import pyplot, pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import loading_and_cleaning.load_dataset

# fixing seed: important to have same random train and test split as the optimizing


np.random.seed(0)


def loading_best_indices_from_second_module(X):
    # loading best indices from the output of the clean_second_module.py
    f = open('best_indices_cvt_auc_recall3.pckl', 'rb')
    best_indices = pickle.load(f)
    f.close()
    indices_new = []
    for j in range(len(best_indices)):
        for k in range(len(best_indices[j])):
            indices_new.append(list(best_indices[j][k]))
    # indices_new1 = np.unique(np.concatenate(indices_new))
    # indices_new2 = list(indices_new1)

    return indices_new

def loading_best_indices_from_DL():
    indices_from_DL_FP = pd.read_csv('optimal_subset.csv')
    # print(indices_from_DL_FP.iloc[0, :])
    indices_from_DL_FP_list = []
    for i in range(indices_from_DL_FP.shape[1]):
        indices_from_DL_FP_list.append(int(indices_from_DL_FP.iloc[0, i]))
    print(len(indices_from_DL_FP_list))
    return list(indices_from_DL_FP_list)

def all_results_SVM(XX_train, YY_train, XX_validation, YY_validation, indices):
    classifier = svm.SVC(probability=True, random_state=3, kernel='linear', C=1.5, class_weight='balanced')
    classifier.fit(XX_train[:, indices], YY_train)
    ts_score = classifier.predict_proba(XX_validation[:, indices])
    return ts_score[:, 1]

def draw_roc_curve(x, y, cv, indices_new, title='ROC Curve'):
    y_real = []
    y_proba = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    counter = -1
    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]
        counter = counter + 1
        if (len(indices_new[counter])):
            # Train
            pred_proba = all_results_SVM(X_train, Y_train, X_train, Y_train, indices_new[counter])
            tot_average_precisionTR.append(average_precision_score(Y_train, pred_proba))
            tot_average_AUCTR.append(roc_auc_score(Y_train, pred_proba))
            y_real.append(Y_train)
            y_proba.append(pred_proba)

            fpr, tpr, thresholds = roc_curve(Y_train, pred_proba)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('Train ROC-AUC')
    plt.close()

def draw_roc_curve_dev(x, y, cv, indices_new, title='ROC Curve for Dev Set'):
    y_real = []
    y_proba = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    counter = -1
    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]
        counter = counter + 1
        if (len(indices_new[counter])):
            # Dev
            pred_proba = all_results_SVM(X_train, Y_train, X_test, Y_test, indices_new[counter])
            tot_average_precisionDev.append(average_precision_score(Y_test, pred_proba))
            tot_average_AUCDev.append(roc_auc_score(Y_test, pred_proba))
            y_real.append(Y_test)
            y_proba.append(pred_proba)
            fpr, tpr, thresholds = roc_curve(Y_test, pred_proba)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('Dev ROC-AUC.png')
    plt.show()
    plt.close()


def draw_roc_curve_Test(x, y, cv, indices_new, title='Test ROC Curve'):
    y_real = []
    y_proba = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    counter = -1
    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]
        counter = counter + 1
        if (len(indices_new[counter])):
            # Train
            pred_proba = all_results_SVM(X_train, Y_train, x_cv, y_cv, indices_new[counter])
            tot_average_precisionTS.append(average_precision_score(y_cv, pred_proba))
            tot_average_AUCTS.append(roc_auc_score(y_cv, pred_proba))
            y_real.append(Y_train)
            y_proba.append(pred_proba)

            fpr, tpr, thresholds = roc_curve(y_cv, pred_proba)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

    # plt.plot(fpr, tpr, lw=1, alpha=0.3,
    #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('Test ROC-AUC.png')
    plt.show()
    plt.close()


def draw_cv_pr_curve(x, y, cv, indices_new, title='Train Precision recall- Curve'):
    y_real = []
    y_proba = []
    counter = -1
    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]
        counter = counter + 1
        if (len(indices_new[counter])):
            # Train
            pred_proba = all_results_SVM(X_train, Y_train, X_train, Y_train, indices_new[counter])
            tot_average_precisionTR.append(average_precision_score(Y_train, pred_proba))
            tot_average_AUCTR.append(roc_auc_score(Y_train, pred_proba))
            y_real.append(Y_train)
            y_proba.append(pred_proba)

            precision, recall, _ = precision_recall_curve(Y_train, pred_proba)

        # # Plotting each individual PR Curve
        # plt.plot(recall, precision, lw=1, alpha=0.3,
        #          label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(Y_train, pred_proba)))


    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)
    no_skill = len(y_real[y_real == 1]) / len(y_real)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('Train Precision recall- Curve.png')
    plt.show()
    plt.close()
def draw_cv_dev_pr_curve(x, y, cv, indices_new, title='Dev Precision recall- Curve'):

    y_real = []
    y_proba = []
    counter = -1
    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]
        counter = counter + 1
        if (len(indices_new[counter])):
            # Train
            pred_proba = all_results_SVM(X_train, Y_train, X_test, Y_test, indices_new[counter])
            y_real.append(Y_test)
            y_proba.append(pred_proba)

            precision, recall, _ = precision_recall_curve(Y_test, pred_proba)

        # # Plotting each individual PR Curve
        # plt.plot(recall, precision, lw=1, alpha=0.3,
        #          label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(Y_train, pred_proba)))


    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    no_skill = len(y_real[y_real == 1]) / len(y_real)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('Dev Precision recall- Curve.png')
    plt.show()
    plt.close()

def draw_cv_test_pr_curve(x, y, cv, indices_new, title='Train Precision recall- Curve'):
    y_real = []
    y_proba = []
    counter = -1
    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        X_train = x[train_index]
        Y_train = y[train_index]
        X_test = x[test_index]
        Y_test = y[test_index]
        counter = counter + 1
        if (len(indices_new[counter])):
            # Train
            pred_proba = all_results_SVM(X_train, Y_train, x_cv, y_cv, indices_new[counter])
            y_real.append(y_cv)
            y_proba.append(pred_proba)
            precision, recall, _ = precision_recall_curve(y_cv, pred_proba)

        # # Plotting each individual PR Curve
        # plt.plot(recall, precision, lw=1, alpha=0.3,
        #          label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(Y_train, pred_proba)))


    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)
    # plt.plot([1, 0], [0, 1], linestyle='--', lw=2, color='r',
    #          alpha=.8)
    no_skill = len(y_real[y_real == 1]) / len(y_real)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('Test Precision recall- Curve.png')
    plt.show()
    plt.close()


def main(X, Y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    indices_new = loading_best_indices_from_second_module(X)
    print("type_old", type(indices_new),len(indices_new),indices_new)
    # indices_new = loading_best_indices_from_DL()
    # print("type_DL", type(indices_new),len(indices_new),indices_new)

    draw_roc_curve(x, y, cv, indices_new, title='ROC Curve (Train)')
    draw_roc_curve_dev(x, y, cv, indices_new, title='ROC Curve (Dev)')
    draw_roc_curve_Test(x, y, cv, indices_new, title='ROC Curve (Test)')
    draw_cv_pr_curve(x, y, cv, indices_new, title='Precision-Recall Curve (Train)')
    draw_cv_dev_pr_curve(x, y, cv, indices_new, title='Precision-Recall Curve (Dev)')
    draw_cv_test_pr_curve(x, y, cv, indices_new, title='Precision-Recall Curve (Test)')
    print(str('Train Average precision: ') + str(np.mean(tot_average_precisionTR) * 100) + str('std: ') + str(
        np.std(tot_average_precisionTR)))
    print(str('Val Average precision: ') + str(np.mean(tot_average_precisionDev) * 100) + str('std: ') + str(
        np.std(tot_average_precisionDev)))
    print(str('Test Average precision: ') + str(np.mean(tot_average_precisionTS) * 100) + str('std: ') + str(
        np.std(tot_average_precisionTS)))

    print(np.mean(tot_average_AUCTR), np.mean(tot_average_AUCDev), np.mean(tot_average_AUCTS))


X,Y = loading_and_cleaning.load_data(sys.argv[1], sys.argv[2])
NUM_TRIALS = 10
counter = -1
indices_ID = range(X.shape[0])
x, x_cv, y, y_cv, indices_x, indices_x_cv = train_test_split(X, Y, indices_ID, test_size=0.2, train_size=0.8,
                                                             stratify=Y, random_state=1)
# Mean Average Precision
tot_average_precisionTR = list()
tot_average_precisionDev = list()
tot_average_precisionTS = list()
# average AUC
tot_average_AUCTR = list()
tot_average_AUCDev = list()
tot_average_AUCTS = list()


main(sys.argv[1], sys.argv[2])
