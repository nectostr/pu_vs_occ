from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from ..base.torchtabular_dataset import TorchtabularDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import torch.utils.data as data
import pickle
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import logging

class Synthetic_Dataset(TorchtabularDataset):

    def __init__(self, root: str, normal_class=0, no_test=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [1]
        # self.outlier_classes.remove(normal_class)

        # TODO: min-max for synth
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

        # min_max = [(0, 10) * 10]
        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
        #                                 transforms.Normalize([min_max[normal_class][0]],
        #                                                      [min_max[normal_class][1] - min_max[normal_class][0]])])
        #
        # target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MySynthetic(root=self.root, train=True)
        # Subset train_set to normal class
        # train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = train_set
        if not no_test:
            self.test_set = MySynthetic(root=self.root, train=False)
        self.train_test_set = MySynthetic(root=self.root, train="train_test")


class MySynthetic(data.Dataset):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root: str, train=True):
        """
        :param root: folder with data to load
        """
        super().__init__()
        self.root = root
        logger = logging.getLogger()
        # TODO: dataset == np array with features & targets
        #  self data = features, self.targets = targets, load with smth
        if train == True:
            with open(os.path.join(self.root,"train.pkl"), "rb") as f:
                full_data = pickle.load(f)

            logger.info(f"train size {len(full_data[full_data[:,-1]==2])}")
            full_data = full_data[full_data[:,-1]==2]

            for i in range(full_data.shape[1] - 2):
                full_data[:, i] = (full_data[:, i] - full_data[:, i].min()) / (
                            full_data[:, i].max() - full_data[:, i].min())

            # if train - all [:,2] going to be 2, but it is zero, known class
            # if test - all [:,2] going to give true answer
            self.data, self.targets = full_data[:,:-2],\
                                       np.zeros(len(full_data)).reshape((-1,1)) #.reshape((full_data.shape[0], full_data.shape[1]-2, 1))
        elif train == False:
            with open(os.path.join(self.root,"test.pkl"), "rb") as f:
                full_data = pickle.load(f)
            # if train - all [:,2] going to be 2, but it is zero, known class
            # if test - all [:,2] going to give true answer
            full_data[:, -1] = np.where(full_data[:, -1] == 2, 0, full_data[:, -1])

            logger.info(f"test size {len(full_data)}, "
                        f"test pos size {len(full_data[full_data[:, -2] == 0])}, "
                        f"test neg size {len(full_data[full_data[:, -2] == 1])}")
            for i in range(full_data.shape[1]-2):
                full_data[:,i] = (full_data[:,i] - full_data[:,i].min())/(full_data[:,i].max() - full_data[:,i].min())



            self.data, self.targets = full_data[:,:-2],\
                                       full_data[:,-1].reshape((-1,1)) #.reshape((full_data.shape[0], full_data.shape[1]-2, 1)),\
        elif train == "train_test":
            with open(os.path.join(self.root,"train.pkl"), "rb") as f:
                full_data = pickle.load(f)

            logger.info(f"train size {len(full_data[full_data[:,-1]==2])}")
            full_data[:, -1] = np.where(full_data[:, -1] == 2, 0, full_data[:, -1])

            for i in range(full_data.shape[1] - 2):
                full_data[:, i] = (full_data[:, i] - full_data[:, i].min()) / (
                            full_data[:, i].max() - full_data[:, i].min())

            # if train - all [:,2] going to be 2, but it is zero, known class
            # if test - all [:,2] going to give true answer
            self.data, self.targets = full_data[:,:-2],\
                                       full_data[:,-1].reshape((-1,1)) #.reshape((full_data.shape[0], full_data.shape[1]-2, 1))

        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).float()

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        source, target = self.data[index], self.targets[index]

        # # TODO: image?!
        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # # img = Image.fromarray(img.numpy(), mode='L')
        #
        # if self.transform is not None:
        #     source = self.transform(source)
        #

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return source, target, index

    def __len__(self):
        return len(self.data)