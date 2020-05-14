import sys

import numpy as np
from scipy.stats import norm, laplace
import random
import os
import pickle
import torch
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, \
    roc_curve, roc_auc_score, brier_score_loss, f1_score

import matplotlib.pyplot as plt
import json

from DEDPUL.algorithms import *
from DEDPUL.utils import *
from DEDPUL.NN_functions import *


def get_train_test_results(root):
    with open(os.path.join(root, "train.pkl"), "rb") as f:
        data = pickle.load(f)
    
    # Print statistics
    # print(f"pos_size {len(data[data[:,-2]==0])}, mix_size {len(data[data[:,-2]==1])}, "
    #       f"total_size {len(data)}, \n"
    #       f"alpha {len(data[data[:,-1] == 1]) / len(data[data[:,-1] != 2])}")
    # all ones still ones, some zeros now ones
    
    # Count current alpha
    alpha = len(data[data[:,-1] == 1]) / len(data[data[:,-1] != 2])
    
    # Normalization
    for i in range(data.shape[1] - 2):
        data[:, i] = (data[:, i] - data[:, i].min()) / (
                data[:, i].max() - data[:, i].min())
    
    # print([(data[:,i].min(),data[:,i].max()) for i in range(data.shape[1])])
    x_train = data[:,:-2]
    y_train = data[:,-2]
    y_true = data[:,-1]
    
    predicted_alpha, poster, net = estimate_poster_cv(x_train, y_train, estimator='dedpul', alpha=None,
                                            estimate_poster_options={'disp': False},
                                            estimate_diff_options={},
                                            estimate_preds_cv_options={
                                                'cv': 3,
                                                'n_networks': 1,
                                                'lr': 0.005,
                                                'hid_dim': 4,
                                                'n_hid_layers': 2
                                            },
                                            train_nn_options={
                                                'n_epochs':10},
                                            get_non_t_class=True
                                            )
    
    # # # Comparing alphas. See further in DEDPUL options.
    # print('predicted alpha:', predicted_alpha, '\nerror:', abs(predicted_alpha - alpha))
    data2 = data.copy()

    j = 0
    for i in range(len(data)):
        if data2[i,-2] == 1:
            data2[i, -2] = poster[j]
            j += 1

    y_true = y_true % 2

    labels, scores = y_true, data2[:, -2]



    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    scores_bin = np.where(scores > optimal_threshold, 1, 0)
    pu_precision_train = precision_score(labels, scores_bin)
    pu_recall_train = recall_score(labels, scores_bin)
    pu_auc_train = roc_auc_score(labels, scores)
    pu_f1_train = f1_score(labels, scores_bin)

    plt.plot(fpr, tpr,)
    plt.title("PU train")
    plt.show()
    pu_precision_train = max(pu_precision_train,
                             precision_score(y_true[y_train == 1], poster.round()))
    pu_recall_train = max(pu_recall_train,
                          recall_score(y_true[y_train == 1], poster.round()))
    pu_auc_train = roc_auc_score(y_true[y_train == 1], poster)
    pu_f1_train = max(pu_f1_train,
                      f1_score(y_true[y_train == 1], poster.round()))

    # TODO think about fullfill it here
    scores_train_pu, labels_train_pu = poster, y_true[y_train == 1]
    # print("Pre", pu_precision_train)
    # print("Rec", pu_recall_train)
    # print("Auc", pu_auc_train)
    # print("f1", pu_f1_train)
    
    
    # TODO PU TEST
    with open(os.path.join(root, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    # print(f"pos_size {len(test_data[test_data[:,-2]==0])},"
    #       f" mix_size {len(test_data[test_data[:,-2]==1])}, "
    #       f"total_size {len(test_data)}, \n"
    #       f"alpha {len(test_data[test_data[:,-1] == 1]) / len(test_data[test_data[:,-1] != 2])}")
    
    true_alpha_in_test = len(test_data[test_data[:,-1] == 1])\
                    / len(test_data[test_data[:,-1] != 2])
    
    # Normalize test data
    for i in range(test_data.shape[1] - 2):
        test_data[:, i] = (test_data[:, i] - test_data[:, i].min()) / (
                test_data[:, i].max() - test_data[:, i].min())
    
    x_test = test_data[:,:-2]
    y_test_train = test_data[:,-2]
    y_test_true: np.ndarray = test_data[:,-1]
    
    #Get test result
    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    outs = net(x_test) * predicted_alpha
    
    outs = outs.detach().numpy()
    y_test_true = y_test_true % 2
    
    labels, scores = y_test_true, outs

    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    scores_bin = np.where(scores > optimal_threshold, 1, 0)
    pu_precision_test = precision_score(labels, scores_bin)
    pu_recall_test = recall_score(labels, scores_bin)
    pu_auc_test = roc_auc_score(labels, scores)
    pu_f1_test = f1_score(labels, scores_bin)

    plt.plot(fpr, tpr,)
    plt.title("PU test")
    plt.show()
    pu_precision_test = max(pu_precision_test,
                            precision_score(y_test_true, outs.round()))
    pu_recall_test = max(pu_recall_test,
                         recall_score(y_test_true, outs.round()))
    pu_auc_test = roc_auc_score(y_test_true, outs)
    pu_f1_test = max(pu_f1_test,
                     f1_score(y_test_true, outs.round()))

    
    scores_test_pu, labels_test_pu = outs, y_test_true
    # print("Pre", pu_precision_test)
    # print("Rec", pu_recall_train)
    # print("Auc", pu_auc_test)
    # print("f1", pu_f1_test)

    return [(scores_train_pu, labels_train_pu,
             (pu_precision_train, pu_recall_train, pu_auc_train, pu_f1_train)),
            (scores_test_pu, labels_test_pu,
             (pu_precision_test, pu_recall_test, pu_auc_test, pu_f1_test))]
    
    
