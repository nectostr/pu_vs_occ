from one_class.Deep_SVDD_PyTorch.src.networks import Synth_Net_Autoencoder
from one_class.Deep_SVDD_PyTorch.src.datasets.main import load_dataset
from one_class.Deep_SVDD_PyTorch.src.optim.ae_trainer import  AETrainer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = torch.device("cuda")
dataset = load_dataset("synth", r"L:\Documents\PyCharmProjects\pu_vs_oc\DATA\synthetic", 0)

ae_net = Synth_Net_Autoencoder(dataset.train_set.data.shape[1])

ae_trainer = AETrainer(lr=0.01, n_epochs=10,
                            batch_size=128, device=device)
ae_net = ae_trainer.train(dataset, ae_net)

ae_trainer.test(dataset, ae_net)
# _, test_loader = dataset.loaders(batch_size=1)
# ae_net.eval()
# with torch.no_grad():
#     for data in test_loader:
#         inputs, labels, idx = data
#         inputs = inputs.to(device)
#         outputs = ae_net(inputs)
#         print(inputs, outputs)