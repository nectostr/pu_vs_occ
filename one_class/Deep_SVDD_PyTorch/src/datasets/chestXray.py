from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
import json

import torch
import torchvision.transforms as transforms


class XRay_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = 0
        self.outlier_classes = [1,]

        # Pre-computed min and max values (after applying GCN) from train data per class
        # min_max = [(-0.8826567065619495, 9.001545489292527),
        #            (-0.6661464580883915, 20.108062262467364),
        #            (-0.7820454743183202, 11.665100841080346),
        #            (-0.7645772083211267, 12.895051191467457),
        #            (-0.7253923114302238, 12.683235701611533),
        #            (-0.7698501867861425, 13.103278415430502),
        #            (-0.778418217980696, 10.457837397569108),
        #            (-0.7129780970522351, 12.057777597673047),
        #            (-0.8280402650205075, 10.581538445782988),
        #            (-0.7369959242164307, 10.697039838804978)]


        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor()
                                        ])
                                        #transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        #transforms.Normalize([min_max[normal_class][0]],
                                        #                     [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyXRay(root=self.root, train=True,
                            transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        # train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, (0,))
        self.train_set = train_set

        self.test_set = MyXRay(root=self.root, train=False,
                                transform=transform, target_transform=target_transform)

# наследования
class MyXRay(VisionDataset):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train, transform, target_transform):
        # TODO: not shure about inheritance. Do I need it at all
        super().__init__(root)

        self.train = train

        if train:
            train_lbl = "train"
        else:
            train_lbl = "test"
        path = os.path.join(os.path.join(root, train_lbl),"labels_paths.jsn")

        with open(path, "r") as f:
            data = json.load(f)
        if train:
            self.train_data = data
        else:
            self.test_data = data

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        #Get butch?
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index][0], self.train_data[index][1]
        else:
            img, target = self.test_data[index][0], self.test_data[index][1]

        img = Image.open(os.path.normpath(os.path.join(self.root, img)))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(np.asarray(img)[:,:,1], mode='L')
        img = np.asarray(img)[:,:,0]
        img = img / 255.0       # before: uint8 and 0..255, after: np.float and 0..1
        img = np.array(img.reshape((1,)+img.shape), dtype=np.float32)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
