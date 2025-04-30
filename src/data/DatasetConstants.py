"""Created by Constantin Philippenko, 12th May 2022."""

import torchvision
import torch.nn as nn
from torch.nn import MSELoss
from torchvision.transforms import transforms

from src.optim.CustomLoss import DiceLoss, CoxLoss, L1WeightedAccuracyLoss
from src.optim.Metric import dice, auc, c_index, accuracy, l1_accuracy, mse_accuracy
from src.optim.nn.Nets import LinearRegression, LogisticRegression, TcgaRegression, CNN_CIFAR10, CNN_MNIST, \
    HeartDiseaseRegression, GoogleNetTransferLearning, LiquidAssetRegression, SynthDataRegression, \
    Synth2ClientsRegression, Synth100ClientsRegression
from src.optim.nn.Unet import UNet

PCA_NB_COMPONENTS = 16

TRANSFORM_MNIST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        # We reshape mnist to match with our neural network
        # ReshapeTransform((-1,))

    ])

TRANSFORM_TRAIN_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

TRANSFORM_TEST_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

MODELS = {"mnist": CNN_MNIST, "mnist_iid": CNN_MNIST, "cifar10": GoogleNetTransferLearning, "cifar10_iid": GoogleNetTransferLearning,
          "heart_disease": HeartDiseaseRegression,
          "tcga_brca": TcgaRegression, "ixi": UNet, "liquid_asset": LiquidAssetRegression, "synth": Synth2ClientsRegression,
          "synth_complex":  Synth100ClientsRegression,
          "exam_llm": None}
CRITERION = {"mnist": nn.CrossEntropyLoss, "mnist_iid": nn.CrossEntropyLoss, "cifar10": nn.CrossEntropyLoss, "cifar10_iid": nn.CrossEntropyLoss,
             "heart_disease": nn.BCELoss,
             "tcga_brca": CoxLoss, "ixi": DiceLoss, "liquid_asset": L1WeightedAccuracyLoss, "synth": MSELoss,
             "synth_complex": MSELoss,
             "exam_llm": nn.CrossEntropyLoss}
METRIC =  {"mnist": accuracy, "mnist_iid": accuracy, "cifar10": accuracy, "cifar10_iid": accuracy, "heart_disease": auc, "tcga_brca": c_index, "ixi": dice,
           "liquid_asset": l1_accuracy, "synth": mse_accuracy, "synth_complex": mse_accuracy,
           "exam_llm": accuracy}

CHECKPOINT = "distilbert-base-uncased" #"bert-base-uncased"

### TODO : Il y a un problème avec la CoxLoss.

TRANSFORM_TRAIN = {"mnist": TRANSFORM_MNIST, "mnist_iid": TRANSFORM_MNIST, "cifar10": TRANSFORM_TRAIN_CIFAR10, "cifar10_iid": TRANSFORM_TRAIN_CIFAR10,
                   "liquid_asset": None}
TRANSFORM_TEST= {"mnist": TRANSFORM_MNIST, "mnist_iid": TRANSFORM_MNIST, "cifar10": TRANSFORM_TEST_CIFAR10, "cifar10_iid": TRANSFORM_TEST_CIFAR10,
                 "liquid_asset": None}

NB_CLIENTS = {"mnist": 20, "mnist_iid": 20, "fashion_mnist": 8, "cifar10": 20, "cifar10_iid": 20, "camelyon16": 2, "heart_disease": 4, "isic2019": 6,
              "ixi": 3, "kits19": 6, "lidc_idri": 5, "tcga_brca": 6, "liquid_asset": 100, "synth": 20,
              "synth_complex": 20,
              "exam_llm": 3}
STEP_SIZE = {"mnist": 0.1, "mnist_iid": 0.1, "cifar10": 0.1, "cifar10_iid": 0.1, "tcga_brca": .015, "heart_disease": 0.1,
             "ixi": 0.01, "liquid_asset": 0.001, "synth": None, "synth_complex": None,
             "exam_llm": 0.0008518845025208505}
WEIGHT_DECAY = {"mnist": 0, "mnist_iid": 0, "cifar10": 0.001, "cifar10_iid": 0.001, "tcga_brca": 0, "heart_disease": 0,
                "ixi": 0, "liquid_asset": 0.1, "synth": 0, "synth_complex": 0, "exam_llm": 0.1}
BATCH_SIZE = {"mnist": 64, "mnist_iid": 128, "cifar10": 64, "cifar10_iid": 128, "tcga_brca": 8, "heart_disease": 8, "ixi": 32, "liquid_asset": 32,
              "synth": 1, "synth_complex": 1, "exam_llm": 32}
MOMENTUM = {"mnist": 0., "mnist_iid": 0., "cifar10": 0.95, "cifar10_iid": 0.95, "tcga_brca": 0, "heart_disease": 0, "ixi": 0.95,
            "liquid_asset": 0.95, "synth": 0, "synth_complex":0, "exam_llm": 0.95}
SCHEDULER_PARAMS = {"mnist": (10, 0.9), "mnist_iid": (4, 0.9), "cifar10": (10, 0.9), "cifar10_iid": (4, 0.9), "tcga_brca": (50, 0.609), "heart_disease": (3, 2/3),
                    "ixi": (4, 0.75), "liquid_asset": (4, 0.75), "synth": (200, 2/3), "synth_complex": (100, 0.5839331768799928),
                    "exam_llm": (15, 2/3)}

# SYNTH COMPLEX
# Best trial config:  {'step_size': 0.012138813439941028, 'weight_decay': 0.001}
# Best trial final validation accuracy:  -5.350926399230957

# HEART_DISEASE
# [I 2025-03-31 16:57:40,443] Trial 1 finished with value: 0.5986238121986389 and parameters: {'step_size': 0.018873042065018778, 'weight_decay': 0.1}. Best is trial 1 with value: 0.5986238121986389.
# [I 2025-03-31 18:42:47,362] Trial 2 finished with value: 0.6494265794754028 and parameters: {'step_size': 0.008349135315129853, 'weight_decay': 0}. Best is trial 2 with value: 0.6494265794754028.
# [I 2025-03-31 18:39:47,888] Trial 1 finished with value: 0.5959862470626831 and parameters: {'step_size': 0.00022183630278209968, 'weight_decay': 0.1}. Best is trial 1 with value: 0.5959862470626831.

# MNIST
# Best trial config:  {'step_size': 0.04500608326765825, 'weight_decay': 0, 'scheduler_gamma': 0.9254090508955067}
# Best trial final validation accuracy:  0.9909150004386902
# IXI
# Best trial config:  {'step_size': 0.019674439294335523, 'weight_decay': 0.001, 'scheduler_gamma': 0.6762453175771046}
# Best trial final validation accuracy:  0.9775788187980652
# EXAM LLM
# Trial 0 finished with value: 0.3566683828830719 and parameters:
# {'step_size': 0.0006362911206636467, 'weight_decay': 0.01, 'scheduler_gamma': 0.9861288300077158}.






