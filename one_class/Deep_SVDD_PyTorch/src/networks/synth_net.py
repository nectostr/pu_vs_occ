import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.base_net import BaseNet


class Synth_Net(BaseNet):
    # TODO: not shure about input dim size, check DEDPUL
    def __init__(self, input_dim=1):
        super().__init__()

        self.rep_dim = 4

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, self.rep_dim)


    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class Synth_Net_Autoencoder(BaseNet):

    def __init__(self, input_dim=1):
        super().__init__()

        self.rep_dim = 4

        # Encoder (must match the Deep SVDD network above)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, self.rep_dim)
        self.fclogvar = nn.Linear(input_dim, self.rep_dim)
        # Decoder
        self.dec1 = nn.Linear(self.rep_dim, input_dim)
        self.dec2 = nn.Linear(input_dim, input_dim)
        # self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fclogvar.weight)
        torch.nn.init.xavier_uniform(self.dec1.weight)
        torch.nn.init.xavier_uniform(self.dec2.weight)

    def encoder(self, x):
        x = F.leaky_relu(self.fc1(x))
        return F.leaky_relu(self.fc2(x)), F.leaky_relu(self.fclogvar(x))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        self.mu, self.logvar = self.encoder(x)
        x = self.reparameterize(self.mu, self.logvar)
        x = F.leaky_relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x

    def get_loss(self, inputs, outputs):
            BCE = F.binary_cross_entropy(outputs, inputs, reduction='mean')

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())

            return BCE + KLD