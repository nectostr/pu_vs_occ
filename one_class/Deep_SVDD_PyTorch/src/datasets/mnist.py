from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST, VisionDataset
from ..base.torchvision_dataset import TorchvisionDataset
import torch
import os
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
import collections

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        self.root = root
        super().__init__(root)


        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class]) \
        if not hasattr(normal_class, "__iter__") else normal_class
        self.outlier_classes = list(range(0, 10))
        for i in self.normal_classes:
            self.outlier_classes.remove(i)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = MyMNIST(root=self.root, train=True,
                            transform=transform, target_transform=target_transform)
        self.test_set = MyMNIST(root=self.root, train=False,
                                transform=transform, target_transform=target_transform)
        self.train_test_set = MyMNIST(root=self.root, train="train_test",
                            transform=transform, target_transform=target_transform)


class MyMNIST(VisionDataset):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train,
                            transform, target_transform, **kwargs):
        super(MyMNIST, self).__init__(root,
                            transform=transform, target_transform=target_transform)
        self.train = train
        if train:
            data_file = "training.pt"
            self.data, self.targets, self.targets_true = torch.load(os.path.join(self.root, data_file))
            if train == True:
                self.data = self.data[self.targets == 0]
                self.targets = self.targets_true[self.targets == 0]
            else:
                self.targets = self.targets_true
            self.train_data = self.data
            self.train_labels = self.targets
        else:
            data_file = "test.pt"
            self.data, self.targets = torch.load(os.path.join(self.root, data_file))

            self.test_data = self.data
            self.test_labels = self.targets





    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed

    def __len__(self):
        return len(self.data)
