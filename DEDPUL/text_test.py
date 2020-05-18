import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import random

from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score, brier_score_loss, f1_score

import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from IPython import display

from DEDPUL.algorithms import *
from DEDPUL.utils import *
from DEDPUL.KMPE import *
from DEDPUL.NN_functions import *

import warnings
warnings.filterwarnings('ignore')


def compare(root):
    df_torch = pd.read_csv(os.path.join(root, 'train.csv'))

    alpha = len(df_torch[(df_torch["label"] == 1) & (df_torch["label_true"] == 1)]) / len(df_torch[df_torch["label"] == 1])
    # Usual DEDPUL function with added "text" modifier as True
    test_alpha, poster, net = estimate_poster_cv(df_torch[["label", "text"]], df_torch["label"], estimator='dedpul', alpha=None,
                                             estimate_poster_options={'disp': False},
                                             estimate_diff_options={},
                                             estimate_preds_cv_options={
                                                 'cv': 3, 'n_networks': 10, 'lr': 0.0005, 'hid_dim': 64,
                                                 'n_hid_layers': 1, 'random_state': 0#,
                                                 , 'text': True
                                             },
                                            train_nn_options={
                                                'n_epochs': 20},
                                            get_non_t_class = True
                                           )

    # Comparing alphas. See further in DEDPUL options.
    print('predicted alpha:', test_alpha, '\nerror:', abs(test_alpha - alpha))

    df_torch["predicted"] = [1 for i in range(len(df_torch))]
    df_torch[df_torch["label"] == 1]["predicted"] = poster
    print(df_torch[["label_true","predicted"]])

    scores = poster
    labels = df_torch[df_torch["label"]==1]['label_true']


    fpr, tpr, threshold = roc_curve(labels, scores)

    plt.plot(fpr, tpr, )
    plt.title("PU text test")
    plt.show()

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    scores_bin = np.where(scores > optimal_threshold, 1, 0)
    pu_precision_train = precision_score(labels, scores_bin)
    pu_recall_train = recall_score(labels, scores_bin)
    pu_auc_train = roc_auc_score(labels, scores)

    print(pu_precision_train)
    print(pu_recall_train)
    print(pu_auc_train)

    # net.

if __name__ == '__main__':
    compare(r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\text_test")