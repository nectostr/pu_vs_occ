import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import random

from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss, f1_score

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from IPython import display

from DEDPUL.algorithms import *
from DEDPUL.utils import *
from DEDPUL.KMPE import *
from DEDPUL.NN_functions import *

import warnings
warnings.filterwarnings('ignore')



# Prepare ususal data
df = pd.read_csv(r"../DATA/text_test/malicious_posts.csv", header = None)
df.columns = "index", "text", "label1", "label2"
df["text"] = (df["text"]
    .str
    .lower()
    .replace(to_replace ='[^A-Za-zА-Яа-я ]+', value = ' ', regex = True)
    .replace(to_replace =' +', value = ' ', regex = True))
df.drop(df[df['text'].map(len) < 10].index, inplace=True)
label_map = {j: i for i,j in enumerate(set(df["label1"]))}
df_torch = pd.DataFrame()
df_torch["label"] = df["label1"].apply(lambda x: label_map[x])
df_torch["text"] = df["text"]

# Cutting data results to two
df_torch.drop(df_torch[(df_torch["label"] != 0) & (df_torch["label"] != 1)].index, inplace=True)

# Mixing data (not the best way)
df_torch["label_true"] = df_torch["label"]
# print(len(df_torch[df_torch["label"] == 0]), len(df_torch[df_torch["label"] == 1]))
df_torch["label"] = df_torch["label"] | np.random.randint(0,2, len(df_torch))
# print(len(df_torch[df_torch["label"] == 0]), len(df_torch[df_torch["label"] == 1]))

alpha = len(df_torch[(df_torch["label"] == 1) & (df_torch["label_true"] == 1)]) / len(df_torch[df_torch["label"] == 1])
print(alpha)

test_alpha, poster = estimate_poster_cv(df_torch, df_torch["label"], estimator='dedpul', alpha=None,
                                         estimate_poster_options={'disp': False},
                                         estimate_diff_options={},
                                         estimate_preds_cv_options={
                                             #'fun': estimate_preds_cv,
                                             'cv': 3, 'n_networks': 10, 'lr': 0.0005, 'hid_dim': 64,
                                             'n_hid_layers': 1, 'random_state': 0#,
                                             # 'train_nn_options': {
                                             #     'n_epochs': 100, 'batch_size': 128,
                                             #     'n_batches': 20, 'n_early_stop': 3, 'disp': False
                                             # }
                                             , 'text': True
                                         },
                                            train_nn_options={
                                                'n_epochs': 1000, 'batch_size': 128,
                                                'n_batches': 20, 'n_early_stop': 3, 'disp': False
                                            }
                                        ,text=True
                                       )

print('alpha:', test_alpha, '\nerror:', abs(test_alpha - alpha))
