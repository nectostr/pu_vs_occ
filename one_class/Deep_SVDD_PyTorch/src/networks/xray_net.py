import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class XRAY_Net(BaseNet):

    def __init__(self):
        super().__init__()

        features_lvl1 = 16
        features_lvl2 = 32
        features_lvl3 = 64
        self.rep_dim = 1024

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


class XRAY_Net_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 1024

        features_lvl1 = 16
        features_lvl2 = 32
        features_lvl3 = 64

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, features_lvl1, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(features_lvl1, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(features_lvl1, features_lvl2, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(features_lvl2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(features_lvl2, features_lvl2, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(features_lvl2, features_lvl3, kernel_size=3, stride=1)
        #TODO: get normal formula
        internal_size = 12544

        self.fc1 = nn.Linear(internal_size, self.rep_dim)

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, internal_size)
        self.deconv1 = nn.ConvTranspose2d(features_lvl3, features_lvl2, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(features_lvl2, features_lvl2, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(features_lvl2, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(features_lvl2, features_lvl1, kernel_size=5, stride=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(features_lvl1, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(features_lvl1, 1, kernel_size=5, stride=3)


    def forward(self, x: torch.Tensor):
        inp_shape = x.shape
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        shape = x.shape
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))
        x = x.view(shape)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.deconv3(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.deconv4(x)
        x = torch.tanh(x)

        assert inp_shape == x.shape
        return x
