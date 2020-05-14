import sys
sys.path += r"L:\Documents\PyCharmProjects\pu_vs_oc\one_class\Deep_SVDD_PyTorch\src"

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score,\
    roc_auc_score, roc_curve, brier_score_loss, f1_score

from .datasets.main import load_dataset
from .deepSVDD import DeepSVDD



def get_train_test_results(root):


    dataset_name = "mnist"
    data_path = root
    normal_class = 0
    dataset = load_dataset(dataset_name, data_path, normal_class)

    nu = 0.1
    deep_SVDD = DeepSVDD("one-class", nu)
    deep_SVDD.set_network("mnist_LeNet")

    #TODO train
    #synth synth_net  --n_epochs 10 --lr 0.0001 --pretrain True --ae_n_epochs 5 --ae_lr 0.001
    deep_SVDD.pretrain(dataset,
                               optimizer_name="adam",
                               lr=0.0002,
                               n_epochs=10,
                               lr_milestones=(),
                               batch_size=128,
                               weight_decay=0.00001,
                               device="cuda")
    deep_SVDD.train(dataset,
                        optimizer_name="adam",
                        lr=0.0001,
                        n_epochs=5,
                        lr_milestones=(),
                        batch_size=128,
                        weight_decay=0.001,
                        device="cuda",
                        sheduler= False)

    #TODO train result
    scores, labels = deep_SVDD.test_on_train(dataset, device="cuda")
    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    scores_bin = np.where(scores > optimal_threshold, 1, 0)

    occ_precision_train = precision_score(labels, scores_bin)
    occ_recall_train = recall_score(labels, scores_bin)
    occ_auc_train = roc_auc_score(labels, scores)
    occ_f1_train = f1_score(labels, scores_bin)

    scores_train_occ, labels_train_occ = scores.copy(), labels.copy()

    #TODO test result
    scores, labels = deep_SVDD.test(dataset, device="cuda")

    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    scores_bin = np.where(scores > optimal_threshold, 1, 0)
    occ_precision_test = precision_score(labels, scores_bin)
    occ_recall_test = recall_score(labels, scores_bin)
    occ_auc_test = roc_auc_score(labels, scores)
    occ_f1_test = f1_score(labels, scores_bin)

    scores_test_occ, labels_test_occ = scores.copy(), labels.copy()

    plt.plot(fpr, tpr,)
    plt.show()

    return [(scores_train_occ, labels_train_occ,
            (occ_precision_train, occ_recall_train, occ_auc_train, occ_f1_train)),
            (scores_test_occ, labels_test_occ,
            (occ_precision_test, occ_recall_test, occ_auc_test, occ_f1_test))]

if __name__ == '__main__':
    train, test = get_train_test_results(r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\MyMNIST")
    print(train[2], test[2])