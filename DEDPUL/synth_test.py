import numpy as np
from scipy.stats import norm, laplace
import random
import os
import pickle
import torch
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss, f1_score

import matplotlib.pyplot as plt
import json

from DEDPUL.algorithms import *
from DEDPUL.utils import *
from DEDPUL.KMPE import *
from DEDPUL.NN_functions import *


import warnings
warnings.filterwarnings('ignore')

root = r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\synthetic"


# Prepare ususal data
with open(os.path.join(root, "train.pkl"), "rb") as f:
    data = pickle.load(f)

print(f"pos_size {len(data[data[:,-2]==0])}, mix_size {len(data[data[:,-2]==1])}, "
      f"total_size {len(data)}, \n"
      f"alpha {len(data[data[:,-1] == 1]) / len(data[data[:,-1] != 2])}")
# all ones still ones, some zeros now ones

alpha = len(data[data[:,-1] == 1]) / len(data[data[:,-1] != 2])

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
print('predicted alpha:', predicted_alpha, '\nerror:', abs(predicted_alpha - alpha))
#
print("Pre", precision_score(poster.round(), y_true[y_train == 1]))
print("Rec", recall_score(poster.round(), y_true[y_train == 1]))
print("Auc", roc_auc_score(y_true[y_train == 1], poster))
print("f1", f1_score(poster.round(), y_true[y_train == 1]))


# TODO PU TEST
with open(os.path.join(root, "test.pkl"), "rb") as f:
    test_data = pickle.load(f)
print(f"pos_size {len(test_data[test_data[:,-2]==0])},"
      f" mix_size {len(test_data[test_data[:,-2]==1])}, "
      f"total_size {len(test_data)}, \n"
      f"alpha {len(test_data[test_data[:,-1] == 1]) / len(test_data[test_data[:,-1] != 2])}")

alpha_in_test = len(test_data[test_data[:,-1] == 1])\
                / len(test_data[test_data[:,-1] != 2])

for i in range(test_data.shape[1] - 2):
    test_data[:, i] = (test_data[:, i] - test_data[:, i].min()) / (
            test_data[:, i].max() - test_data[:, i].min())

# print([(data[:,i].min(),data[:,i].max()) for i in range(data.shape[1])])
x_test = test_data[:,:-2]
y_test_train = test_data[:,-2]
y_test_true: np.ndarray = test_data[:,-1]

x_test = torch.as_tensor(x_test, dtype=torch.float32)
outs = net(x_test) * predicted_alpha

# print(outs)
outs = outs.detach().numpy()
y_test_true = y_test_true % 2
print("Pre", precision_score(outs.round(), (y_test_true/2+0.1).round()))
print("Rec", recall_score(outs.round(), (y_test_true/2+0.1).round()))
print("Auc", roc_auc_score((y_test_true/2+0.1).round(), outs))
print("f1", f1_score(outs.round(), (y_test_true/2+0.1).round()))