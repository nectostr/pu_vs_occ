import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from random import sample
import tqdm

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

def logger_start(f):
    def inner(*a, **kw):
        print(f.__name__, "started")
        r = f(*a, **kw)
        print(f.__name__, "finished")
        return r
    return inner


class UsualNet(nn.Module):
    def __init__(self, inp_dim, out_dim=1, hid_dim=32, n_hid_layers=1, bayes=False):
        super(UsualNet, self).__init__()
        self.bayes = bayes
        self.n_hid_layers = n_hid_layers

        self.inp = nn.Linear(inp_dim, hid_dim)
        if self.n_hid_layers > 0:
            self.hid = nn.Sequential()
            for i in range(n_hid_layers):
                self.hid.add_module(str(i), nn.Linear(hid_dim, hid_dim))
                self.hid.add_module('a' + str(i), nn.ReLU())

        if self.bayes:
            self.out_mean = nn.Linear(hid_dim, out_dim)
            self.out_logvar = nn.Linear(hid_dim, out_dim)
        else:
            self.out = nn.Linear(hid_dim, out_dim)


    def forward(self, x, return_params=False, sample_noise=False):
        x = F.relu(self.inp(x))
        if self.n_hid_layers > 0:
            x = self.hid(x)

        if self.bayes:
            mean, logvar = self.out_mean(x), self.out_logvar(x)
            var = torch.exp(logvar * .5)
            if sample_noise:
                x = mean + var * torch.randn_like(var)
            else:
                x = mean
        else:
            mean = self.out(x)
            var = torch.zeros_like(mean) + 1e-3
            x = mean
        p = F.sigmoid(x)

        if return_params:
            return p, mean, var
        else:
            return p

class ConvNet(nn.Module):
    def __init__(self, inp_dim=(32, 32, 3), out_dim=1, hid_dim_full=128, bayes=False):
        super(ConvNet, self).__init__()
        self.bayes = bayes

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, 1)
        self.conv6 = nn.Conv2d(32, 4, 1)

        self.conv_to_fc = 4*56*56
        self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
        if self.bayes:
            self.out_mean = nn.Linear(hid_dim_full, out_dim)
            self.out_logvar = nn.Linear(hid_dim_full, out_dim)
        else:
            self.out = nn.Linear(hid_dim_full, out_dim)


    def forward(self, x, return_params=False, sample_noise=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(-1, self.conv_to_fc)
        x = F.relu(self.fc1(x))

        if self.bayes:
            mean, logvar = self.out_mean(x), self.out_logvar(x)
            var = torch.exp(logvar * .5)
            if sample_noise:
                x = mean + var * torch.randn_like(var)
            else:
                x = mean
        else:
            mean = self.out(x)
            var = torch.zeros_like(mean) + 1e-3
            x = mean
        p = F.sigmoid(x)

        if return_params:
            return p, mean, var
        else:
            return p

def get_discriminator(inp_dim, out_dim=1, hid_dim=32, n_hid_layers=1, bayes=False, net_name=None):
    """
    Feed-forward Neural Network constructor
    :param inp_dim: number of input dimensions
    :param out_dim: number of output dimensions; 1 for binary classification
    :param hid_dim: number of hidden dimensions
    :param n_hid_layers: number of hidden layers

    :return: specified neural network
    """
    if net_name is None:
        return UsualNet(inp_dim, out_dim, hid_dim, n_hid_layers, bayes)
    elif net_name == 'xray_net':
        return XRAY_Net()


def all_convolution(inp_dim=(32, 32, 3), out_dim=1, hid_dim_full=128, bayes=False):
   return ConvNet(inp_dim, out_dim, hid_dim_full, bayes=bayes)


def d_loss_standard(batch_mix, batch_pos, discriminator, loss_function=None):
    d_mix = discriminator(batch_mix)
    d_pos = discriminator(batch_pos)
    if (loss_function is None) or (loss_function == 'log'):
        loss_function = lambda x: torch.log(x + 10 ** -5)  # log loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == 'brier':
        loss_function = lambda x: x ** 2  # brier loss
    return -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2

def KL_normal(m, s):
    return (torch.log(1 / (s + 1e-6)) + (m ** 2 + s ** 2 - 1) * .5).mean()

def d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function=None, w=0.1):
    d_mix, mean_mix, std_mix = discriminator(batch_mix, return_params=True, sample_noise=True)
    d_pos, mean_pos, std_pos = discriminator(batch_pos, return_params=True, sample_noise=True)
    if (loss_function is None) or (loss_function == 'log'):
        loss_function = lambda x: torch.log(x + 10 ** -5)  # log loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == 'brier':
        loss_function = lambda x: x ** 2  # brier loss
    loss = -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2
    loss += (KL_normal(mean_mix, std_mix) + KL_normal(mean_pos, std_pos)) / 2 * w
    return loss

def d_loss_nnRE(batch_mix, batch_pos, discriminator, alpha, beta=0., gamma=1., loss_function=None):
    d_mix = discriminator(batch_mix)
    d_pos = discriminator(batch_pos)
    if (loss_function is None) or (loss_function == 'brier'):
        loss_function = lambda x: (1 - x) ** 2 # brier loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: 1 - x  # sigmoid loss
    elif loss_function in {'log', 'logistic'}:
        loss_function = lambda x: torch.log(1 - x + 10 ** -5) # log loss
    pos_part = (1 - alpha) * torch.mean(loss_function(1 - d_pos))
    nn_part = torch.mean(loss_function(d_mix)) - (1 - alpha) * torch.mean(loss_function(d_pos))

    # return nn_part + pos_part, 1

    if nn_part.item() >= - beta:
        return pos_part + nn_part, 1
    else:
        return -nn_part, gamma

#@logger_start
def train_NN(mix_data, pos_data, discriminator, d_optimizer, mix_data_test=None, pos_data_test=None,
             batch_size=None, n_epochs=10, n_batches=20, n_early_stop=5,
             d_scheduler=None, training_mode='standard', disp=False, loss_function=None, nnre_alpha=None,
             metric=None, stop_by_metric=False, bayes=False, bayes_weight=1e-5):
    """
    Train discriminator to classify mix_data from pos_data.
    """
    d_losses_train = []
    d_losses_test = []
    d_metrics_test = []
    batch_size_mix = int(mix_data.shape[0] / n_batches)
    batch_size_pos = int(pos_data.shape[0] / n_batches)
    if mix_data_test is not None:
        data_test = np.concatenate((pos_data_test, mix_data_test))
        target_test = np.concatenate((np.zeros((pos_data_test.shape[0],)), np.ones((mix_data_test.shape[0],))))
    for epoch in tqdm.tqdm(range(n_epochs)):
        d_losses_cur = []
        if d_scheduler is not None:
            d_scheduler.step()

        for i in range(n_batches):

            batch_mix = np.array(sample(list(mix_data), batch_size_mix))
            batch_pos = np.array(sample(list(pos_data), batch_size_pos))

            batch_mix = torch.as_tensor(batch_mix, dtype=torch.float32)
            batch_pos = torch.as_tensor(batch_pos, dtype=torch.float32)
            batch_mix.requires_grad_(True)
            batch_pos.requires_grad_(True)

            # Optimize D
            d_optimizer.zero_grad()

            if training_mode == 'standard':
                if bayes:
                    loss = d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function, bayes_weight)
                else:
                    loss = d_loss_standard(batch_mix, batch_pos, discriminator, loss_function)
                loss.backward()
                d_optimizer.step()

            elif training_mode == 'nnre':
                loss, gamma = d_loss_nnRE(batch_mix, batch_pos, discriminator, nnre_alpha, beta=0, gamma=1, loss_function=loss_function)

                for param_group in d_optimizer.param_groups:
                    param_group['lr'] *= gamma

                loss.backward()
                d_optimizer.step()

                for param_group in d_optimizer.param_groups:
                    param_group['lr'] /= gamma
            d_losses_cur.append(loss.cpu().item())

        d_losses_train.append(round(np.mean(d_losses_cur), 5))

        if mix_data_test is not None and pos_data_test is not None:
            if training_mode == 'standard':
                if bayes:
                    loss = d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function, bayes_weight)
                else:
                    loss = d_loss_standard(batch_mix, batch_pos, discriminator, loss_function)
                if bayes:
                    d_losses_test.append(round(d_loss_bayes(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                            torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                            discriminator, w=bayes_weight).item(), 5))
                else:
                    d_losses_test.append(round(d_loss_standard(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                               torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                               discriminator).item(), 5))
            elif training_mode == 'nnre':
                d_losses_test.append(round(d_loss_nnRE(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                       torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                       discriminator, nnre_alpha, beta=10)[0].item(), 5))
            if metric is not None:
                d_metrics_test.append(metric(target_test,
                                             discriminator(torch.as_tensor(data_test, dtype=torch.float32)).detach().numpy()))

            if disp:
                if not metric:
                    print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1])
                else:
                    print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1],
                          'test_metric=', d_metrics_test[-1])

            if epoch >= n_early_stop:
                if_stop = True
                for i in range(n_early_stop):
                    if metric is not None and stop_by_metric:
                        if d_metrics_test[-i - 1] < d_metrics_test[-n_early_stop - 1]:
                            if_stop = False
                            break
                    else:
                        if d_losses_test[-i-1] < d_losses_test[-n_early_stop-1]:
                            if_stop = False
                            break
                if if_stop:
                    break
        elif disp:
            print('epoch', epoch, ', train_loss=', d_losses_train[-1])

    return d_losses_train, d_losses_test


def init_keras_model(n_layers=1, n_hid=32, lr=10**-5):
    clf = Sequential()
    for _ in range(n_layers):
        clf.add(Dense(n_hid, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['acc'])
    return clf



class XRAY_Net(nn.Module):

    def __init__(self):
        super().__init__()

        features_lvl1 = 16
        features_lvl2 = 32
        features_lvl3 = 64
        self.rep_dim = 32

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, features_lvl1, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(features_lvl1, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(features_lvl1, features_lvl2, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(features_lvl2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(features_lvl2, features_lvl2, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(features_lvl2, features_lvl3, kernel_size=3, stride=1)
        internal_size = 12544

        self.fc1 = nn.Linear(internal_size, self.rep_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
