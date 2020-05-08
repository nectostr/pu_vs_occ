import os
import pickle


from DEDPUL.algorithms import *

#TODO: load data and prepare
root = r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\synthetic"
with open(os.path.join(root, "train.pkl"), "rb") as f:
    data = pickle.load(f)
print(f"pos_size {len(data[data[:,-2]==0])}, mix_size {len(data[data[:,-2]==1])}, "
      f"total_size {len(data)}, \n"
      f"alpha {len(data[data[:,-1] == 1]) / len(data[data[:,-1] != 2])}")
alpha = len(data[data[:,-1] == 1]) / len(data[data[:,-1] != 2])
for i in range(data.shape[1] - 2):
    data[:, i] = (data[:, i] - data[:, i].min()) / (
            data[:, i].max() - data[:, i].min())
x_train = data[:,:-2]
y_train = data[:,-2]
y_true = data[:,-1]

#TODO: save as "current data"


#TODO OCC part
#TODO load data
#TODO train
#TODO train result
#TODO test result
#TODO save

#TODO PU part
test_alpha, poster, non_t_class, net = estimate_poster_cv(x_train, y_train, estimator='dedpul', alpha=None,
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
                                        get_non_trad = True
                                        )
#TODO test result
print(test_alpha)


#TODO save