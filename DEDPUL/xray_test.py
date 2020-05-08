import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss, f1_score

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from IPython import display
import json

from DEDPUL.algorithms import *
from DEDPUL.utils import *
from DEDPUL.KMPE import *
from DEDPUL.NN_functions import *


import warnings
warnings.filterwarnings('ignore')

root = r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\ChestXRayPneu"


# Prepare ususal data
with open(root + r"\train\labels_paths.jsn", "r") as f:
    data = json.load(f)

# ones = [i for i in data if i[1]==1]
# zeros = [i for i in data if i[1]==0]
# print(len(ones), len(zeros))
img_shape = (224,224)

size = 1000
x_data = np.zeros((size,) + img_shape)
labels = np.full(size, 0, dtype=int)
ind = 0
for path, lbl in data:
    if ind < size//2 and lbl == 1 or ind > size//2 and lbl==0:
        continue
    if ind > size-1:
        break
    try:
        x_data[ind] = np.asarray(Image.open(os.path.join(root,path)))[:,:,0]
        labels[ind] = lbl
        ind += 1
    except:
        print(f"{path} go wrong")

x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.01)
# print(set(y_train))
alpha = 0.8

neg_mix_size = (y_train == 1).sum()
mix_size = (neg_mix_size/alpha)
pos_mix_size = mix_size*(1-alpha)

pos_size = (y_train == 0).sum() - pos_mix_size

print(f"pos_size {pos_size}, mix_size {mix_size}, total_size {len(y_train)}, \n"
      f"pos_mix_size {pos_mix_size}, neg_mix_size {neg_mix_size}, in total {mix_size}\n"
      f"alpha{neg_mix_size/mix_size}")
# all ones still ones, some zeros now ones

y_train_true = y_train.copy()
i = 0
while (y_train == 1).sum() < mix_size:
    ind = np.random.randint(len(y_train))
    if y_train[ind] == 0:
        y_train[ind] = 1
        i += 1

print(f"pos size {(y_train == 0).sum()}, mix_size {(y_train == 1).sum()}")

x_train = x_train.reshape((x_train.shape[0],)+(1,) + x_train.shape[1:])
print(x_train.shape)
#Usual DEDPUL function with added "text" modifier as True
test_alpha, poster = estimate_poster_cv(x_train, y_train, estimator='dedpul', alpha=None,
                                         estimate_poster_options={'disp': False},
                                         estimate_diff_options={},
                                         estimate_preds_cv_options={
                                             "all_conv": True
                                         }
                                       )

# # Comparing alphas. See further in DEDPUL options.
print('predicted alpha:', test_alpha, '\nerror:', abs(test_alpha - alpha))

print(precision_score(poster.round(), y_train_true[y_train == 1]))
print(recall_score(poster.round(), y_train_true[y_train == 1]))
print(roc_auc_score(y_train_true[y_train == 1], poster))
print(f1_score(poster.round(), y_train_true[y_train == 1]))