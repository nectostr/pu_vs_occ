import sys
sys.path += r"L:\Documents\PyCharmProjects\pu_vs_oc\one_class\Deep_SVDD_PyTorch\src"

import click
import torch
import logging
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
# TODO: name of dataset
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'synth', 'xray224']))
# TODO: net name
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet',
                                               'cifar10_LeNet_ELU', 'synth_net', 'xray_Net']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.debug('Log file is %s.' % log_file)
    logger.debug('Data path is %s.' % data_path)
    logger.debug('Export path is %s.' % xp_path)

    logger.debug('Dataset: %s' % dataset_name)
    logger.debug('Normal class: %d' % normal_class)
    logger.debug('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.debug('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.debug('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.debug('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.debug('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.debug('Computation device: %s' % device)
    logger.debug('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    # TODO: 1. Load Data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # TODO: 2. Init model
    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    if net_name == "synth_net":
        deep_SVDD.set_network(net_name, dataset.train_set.data.shape[1])
    else:
        deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        # отсебятина, что бы прогрузить только предтраиненый автоэнкодер
        # deep_SVDD.load_model(model_path=load_model, load_ae=True)
        # logger.debug('Loading model from %s.' % load_model)
    # if load_ae:
        deep_SVDD.load_ae_model(load_model)

    logger.debug('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.debug('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.debug('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.debug('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.debug('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.debug('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.debug('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # TODO: 3. Pretrain
        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.debug('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.debug('Training learning rate: %g' % cfg.settings['lr'])
    logger.debug('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.debug('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.debug('Training batch size: %d' % cfg.settings['batch_size'])
    logger.debug('Training weight decay: %g' % cfg.settings['weight_decay'])

    # TODO: 4. Train
    # Train model on dataset

    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    sheduler=not net_name=="synth_net")

    # TODO: 4. Test
    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[(labels == 0).reshape(indices.shape)][np.argsort(scores[(labels == 0).reshape(scores.shape)])]  # sorted from lowest to highest anomaly score

    if dataset_name in ('mnist', 'cifar10'):

        if dataset_name == 'mnist':
            X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

        elif dataset_name == 'cifar10':
            X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

        elif dataset_name == 'xray224':
            X_normals = dataset.test_set.data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.data[idx_sorted[-32:], ...].unsqueeze(1)

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + '/model.tar', save_ae=False)
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
