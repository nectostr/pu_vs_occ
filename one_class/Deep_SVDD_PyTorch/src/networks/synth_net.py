import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class Synth_Net(BaseNet):
    # TODO: not shure about input dim size, check DEDPUL
    def __init__(self, input_dim=1):
        super().__init__()

        self.rep_dim = 1


        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, input_dim*2)
        self.fc3 = nn.Linear(input_dim * 2, input_dim)
        self.fc4 = nn.Linear(input_dim, self.rep_dim)


    def forward(self, x: torch.Tensor):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class Synth_Net_Autoencoder(BaseNet):

    def __init__(self, input_dim=1):
        super().__init__()

        self.rep_dim = 1
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, input_dim * 2)
        self.fc3 = nn.Linear(input_dim * 2, input_dim)
        self.fc4 = nn.Linear(input_dim, self.rep_dim)

        # Decoder
        self.fc11 = nn.Linear(self.rep_dim, input_dim)
        self.fc21 = nn.Linear(input_dim, input_dim * 2)
        self.fc31 = nn.Linear(input_dim * 2, input_dim * 2)
        self.fc41 = nn.Linear(input_dim * 2, input_dim)




    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc11(x))
        x = F.leaky_relu(self.fc21(x))
        x = F.leaky_relu(self.fc31(x))
        x = F.leaky_relu(self.fc41(x))
        x = torch.sigmoid(x)
        return x
