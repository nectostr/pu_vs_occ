from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .synthetic import Synthetic_Dataset
from .chestXray import XRay_Dataset

def load_dataset(dataset_name, data_path, normal_class, in_memory=False):
    """Loads the dataset."""

    # TODO: added synth
    implemented_datasets = ('mnist', 'cifar10', 'synth', 'xray224')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    elif dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    elif dataset_name == 'synth':
        dataset = Synthetic_Dataset(root=data_path, normal_class=normal_class)

    elif dataset_name == 'xray224':
        dataset = XRay_Dataset(root=data_path, normal_class=normal_class, in_memory=in_memory)
    return dataset
