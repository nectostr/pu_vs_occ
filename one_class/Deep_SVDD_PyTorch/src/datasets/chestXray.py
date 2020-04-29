from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
import json
import logging
import torch
import torchvision.transforms as transforms


class XRay_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0, in_memory=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = 0
        self.outlier_classes = [1,]



        train_set = MyXRay(root=self.root, train=True, in_memory=True)
        # Subset train_set to normal class
        # train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, (0,))
        # self.train_set = train_set

        self.test_set = MyXRay(root=self.root, train=False, in_memory=True)

# наследования
class MyXRay(VisionDataset):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train, in_memory=False):
        # TODO: not shure about inheritance. Do I need it at all
        super().__init__(root)
        self.train = train
        self.in_memory = in_memory
        if train:
            train_lbl = "train"
        else:
            train_lbl = "test"
        path = os.path.join(os.path.join(root, train_lbl),"labels_paths.jsn")

        with open(path, "r") as f:
            data = json.load(f)

        i = 0
        while i < len(data):
            # TODO: Replace - replace it in real
            if not os.path.exists(os.path.join(self.root, data[i][0].replace("\\", "/"))):
                del data[i]
            else:
                i += 1

        logging.info(f"{'train' if train else 'test'} dataset has {len(data)} len")

        if self.in_memory:
            for i in range(len(data)):
                img = Image.open(os.path.normpath(os.path.join(self.root, data[i][0].replace("\\", "/"))))
                img = np.asarray(img)[:, :, 0]
                img = img / 255.0
                img = np.array(img.reshape((1,) + img.shape), dtype=np.float32)
                img = torch.from_numpy(img)
                data[i][0] = img

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

        if not self.in_memory:
            #TODO: real replce
            img = Image.open(os.path.normpath(os.path.join(self.root, img.replace("\\", "/"))))

            img = np.asarray(img)[:,:,0]
            img = img / 255.0       # before: uint8 and 0..255, after: np.float and 0..1
            img = np.array(img.reshape((1,)+img.shape), dtype=np.float32)
            img = torch.from_numpy(img)

        return img, target, index  # only line changed
