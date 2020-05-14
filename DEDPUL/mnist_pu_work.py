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

    processed_folder = r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\MyMNIST\processed"
    data_file = "training.pt"
    data, targets = torch.load(os.path.join(processed_folder, data_file))

    normal_classes = [0,]
    outlier_classes = list(range(0, 10))
    for i in normal_classes:
        outlier_classes.remove(i)

    data = data.numpy()
    targets = targets.numpy()

    mask = np.isin(targets, normal_classes)
    targets_true = 1 - mask


    neg_mix_size = len(targets_true[targets_true == 1])
    pos_mix_size = len(targets_true[targets_true == 0]) // 2
    alpha = neg_mix_size / (neg_mix_size + pos_mix_size)
    pos_size = len(targets_true[targets_true == 0]) - pos_mix_size

    targets_to_train = targets_true.copy()
    n = 0
    while n < pos_mix_size:
        ind = np.random.randint(0,len(targets_true))
        if targets_to_train[ind] == 0:
            targets_to_train[ind] = 1
            n += 1


    # Print statistics
    print(f"pos_size {len(targets_to_train[targets_to_train ==0])}, "
          f"mix_size {len(targets_to_train[targets_to_train ==1])}, "
          f"total_size {len(targets_to_train)}, \n"
          f"alpha {len(targets_true[targets_true == 1]) / len(targets_to_train[targets_to_train == 1])}")

    #all ones still ones, some zeros now ones
    
    # Count current alpha
    
    # Normalization
    data = np.array(data / 255, dtype=float)
    data = data.reshape((data.shape[0],) + (1,) + data.shape[1:])
    
    # print([(data[:,i].min(),data[:,i].max()) for i in range(data.shape[1])])

    n = np.random.choice(np.arange(0,len(targets_true)),len(targets_true), replace=False)

    data = data[n]
    targets_true = targets_true[n]
    targets_to_train = targets_to_train[n]
    
    predicted_alpha, poster, net = estimate_poster_cv(data, targets_to_train, estimator='dedpul', alpha=None,
                                            estimate_poster_options={'disp': False},
                                            estimate_diff_options={},
                                            estimate_preds_cv_options={
                                                'all_conv': True,
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
    targets_true2 = targets_true.copy()

    j = 0
    for i in range(len(data)):
        if targets_true[i] == 1:
            targets_true2[i] = poster[j]
            j += 1


    labels, scores = targets_true, targets_true2



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
                             precision_score(targets_true[targets_to_train == 1], poster.round()))
    pu_recall_train = max(pu_recall_train,
                          recall_score(targets_true[targets_to_train == 1], poster.round()))
    pu_auc_train = roc_auc_score(targets_true[targets_to_train == 1], poster)
    pu_f1_train = max(pu_f1_train,
                      f1_score(targets_true[targets_to_train == 1], poster.round()))

    # TODO think about fullfill it here
    scores_train_pu, labels_train_pu = poster, targets_true[targets_to_train == 1]
    # print("Pre", pu_precision_train)
    # print("Rec", pu_recall_train)
    # print("Auc", pu_auc_train)
    # print("f1", pu_f1_train)
    
    
    # TODO PU TEST

    data_file = "test.pt"
    data, targets = torch.load(os.path.join(processed_folder, data_file))

    normal_classes = [0, ]
    outlier_classes = list(range(0, 10))
    for i in normal_classes:
        outlier_classes.remove(i)

    targets = targets.numpy()

    mask = np.isin(targets, normal_classes)
    targets_true = 1 - mask
    
    data = np.array(data / 255, dtype=float)
    

    #Get test result
    data = data.reshape((data.shape[0],) + (1,) + data.shape[1:])
    data = torch.as_tensor(data, dtype=torch.float32)
    outs = net(data) * predicted_alpha
    
    outs = outs.detach().numpy()
    
    labels, scores = targets_true, outs

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
                            precision_score(targets_true, outs.round()))
    pu_recall_test = max(pu_recall_test,
                         recall_score(targets_true, outs.round()))
    pu_f1_test = max(pu_f1_test,
                     f1_score(targets_true, outs.round()))

    
    scores_test_pu, labels_test_pu = outs, targets_true
    # print("Pre", pu_precision_test)
    # print("Rec", pu_recall_train)
    # print("Auc", pu_auc_test)
    # print("f1", pu_f1_test)

    return [(scores_train_pu, labels_train_pu,
             (pu_precision_train, pu_recall_train, pu_auc_train, pu_f1_train)),
            (scores_test_pu, labels_test_pu,
             (pu_precision_test, pu_recall_test, pu_auc_test, pu_f1_test))]
    
if __name__ == '__main__':

    train, test = get_train_test_results("")
    print(train[2], test[2])