from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .synth_net import Synth_Net, Synth_Net_Autoencoder
from .xray_net import XRAY_Net, XRAY_Net_Autoencoder

implemented_networks = ('mnist_LeNet', 'cifar10_LeNet',
                        'cifar10_LeNet_ELU', 'synth_net', "xray_Net")

def build_network(net_name, input_dim=None):
    """Builds the neural network."""

    # TODO: Added new implemented network net
    global implemented_networks
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    elif net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    elif net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    elif net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    elif net_name == 'synth_net':
        net = Synth_Net() if input_dim is None else Synth_Net(input_dim)

    elif net_name == 'xray_Net':
        net = XRAY_Net()

    return net


def build_autoencoder(net_name, input_dim=None):
    """Builds the corresponding autoencoder network."""

    global implemented_networks
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    elif net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    elif net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    elif net_name == 'synth_net':
        ae_net = Synth_Net_Autoencoder() \
            if input_dim is None \
            else Synth_Net_Autoencoder(input_dim)

    elif net_name == 'xray_Net':
        ae_net = XRAY_Net_Autoencoder()

    return ae_net
