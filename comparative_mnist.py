import sys

import numpy as np
from scipy.stats import norm, laplace
import random
import os
import pickle
import torch
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss, f1_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json

from DEDPUL.algorithms import *
from DEDPUL.utils import *
from DEDPUL.KMPE import *
from DEDPUL.NN_functions import *

import warnings
warnings.filterwarnings('ignore')


import DEDPUL.mnist_pu_work as pu
import one_class.Deep_SVDD_PyTorch.src.mnist_occ_work as occ


def comare(root):


    n = 20

    occ_precision_train_s, occ_recall_train_s, occ_auc_train_s, occ_f1_train_s = [],[],[],[]
    occ_precision_test_s, occ_recall_test_s, occ_auc_test_s, occ_f1_test_s = [],[],[],[]

    pu_precision_train_s, pu_recall_train_s, pu_auc_train_s, pu_f1_train_s = [],[],[],[]
    pu_precision_test_s, pu_recall_test_s, pu_auc_test_s, pu_f1_test_s = [],[],[],[]
    corrs = []

    for i in range(n):
        print(f"____________STEP {i}_______________")
        print("PU PATR")
        train_pu, test_pu = pu.get_train_test_results(root)
        scores_train_pu, labels_train_pu, (pu_precision_train, pu_recall_train, pu_auc_train, pu_f1_train) = train_pu
        scores_test_pu, labels_test_pu, (pu_precision_test, pu_recall_test, pu_auc_test, pu_f1_test) = test_pu
        pu_precision_train_s.append(pu_precision_train)
        pu_recall_train_s.append(pu_recall_train)
        pu_auc_train_s.append(pu_auc_train)
        pu_f1_train_s.append(pu_f1_train)
        pu_precision_test_s.append(pu_precision_test)
        pu_recall_test_s.append(pu_recall_test)
        pu_auc_test_s.append(pu_auc_test)
        pu_f1_test_s.append(pu_f1_test)

    #TODO OCC part
        print("OCC PATR")
        train_occ, test_occ = occ.get_train_test_results(root)
        scores_train_occ, labels_train_occ, (occ_precision_train, occ_recall_train, occ_auc_train, occ_f1_train) = train_occ
        scores_test_occ, labels_test_occ, (occ_precision_test, occ_recall_test, occ_auc_test, occ_f1_test) = test_occ
        occ_precision_train_s.append(occ_precision_train)
        occ_recall_train_s.append(occ_recall_train)
        occ_auc_train_s.append(occ_auc_train)
        occ_f1_train_s.append(occ_f1_train)
        occ_precision_test_s.append(occ_precision_test)
        occ_recall_test_s.append(occ_recall_test)
        occ_auc_test_s.append(occ_auc_test)
        occ_f1_test_s.append(occ_f1_test)

        cor = pearsonr(scores_test_pu.reshape(-1), scores_test_occ)[0]
        corrs.append(cor)
        print(cor)


    print("___ TRAIN RESULTS _____")

    print("Precision")
    print(f"pu {np.median(pu_precision_train_s)}, occ {np.median(occ_precision_train_s)}")
    print("Recall")
    print(f"pu {np.median(pu_recall_train_s)}, occ {np.median(occ_recall_train_s)}")
    print("AUC")
    print(f"pu {np.median(pu_auc_train_s)}, occ {np.median(occ_auc_train_s)}")
    print("F1 score")
    print(f"pu {np.median(pu_f1_train_s)}, occ {np.median(occ_f1_train_s)}")
    print()

    print("___ TEST RESULTS _____")
    print("Precision")
    print(f"pu {np.median(pu_precision_test_s)}, occ {np.median(occ_precision_test_s)}")
    print("Recall")
    print(f"pu {np.median(pu_recall_test_s)}, occ {np.median(occ_recall_test_s)}")
    print("AUC")
    print(f"pu {np.median(pu_auc_test_s)}, occ {np.median(occ_auc_test_s)}")
    print("F1 score")
    print(f"pu {np.median(pu_f1_test_s)}, occ {np.median(occ_f1_test_s)}")
    print()


    print(np.mean(corrs))

    d = {"train":{"pu_precision_train_s":pu_precision_train_s,
         "occ_precision_train_s":occ_precision_train_s,
         "pu_recall_train_s":pu_recall_train_s,
         "occ_recall_train_s":occ_recall_train_s,
         "pu_auc_train_s":pu_auc_train_s,
         "occ_auc_train_s":occ_auc_train_s,
         "pu_f1_train_s":pu_f1_train_s,
         "occ_f1_train_s":occ_f1_train_s},
         "test":{"pu_precision_test_s":pu_precision_test_s,
         "occ_precision_test_s":occ_precision_test_s,
         "pu_recall_test_s":pu_recall_test_s,
         "occ_recall_test_s":occ_recall_test_s,
         "pu_auc_test_s":pu_auc_test_s,
         "occ_auc_test_s":occ_auc_test_s,
         "pu_f1_test_s":pu_f1_test_s,
         "occ_f1_test_s":occ_f1_test_s},
         "corrs":corrs}

    with open(r"DATA\results\mnist\usual.pkl", "wb") as f:
        pickle.dump(d, f)

if __name__ == '__main__':
    root = r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\MyMNIST\current"
    comare(root)