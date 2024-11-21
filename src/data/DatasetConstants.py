"""Created by Constantin Philippenko, 12th May 2022."""

import torchvision
import torch.nn as nn
from torch.nn import MSELoss
from torchvision.transforms import transforms

from src.optim.CustomLoss import DiceLoss, CoxLoss, L1WeightedAccuracyLoss
from src.optim.Metric import dice, auc, c_index, accuracy, l1_accuracy, mse_accuracy
from src.optim.nn.Nets import LinearRegression, LogisticRegression, TcgaRegression, CNN_CIFAR10, CNN_MNIST, \
    HeartDiseaseRegression, GoogleNetTransferLearning, LiquidAssetRegression, SynthDataRegression
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


# NB_LABELS = {"mnist": 10, "fashion_mnist": 10, "cifar10": 10,
#              "camelyon16": 2,  "heart_disease": 2, "isic2019": 8,
#              "ixi": None, "kits19": None, "lidc_idri": None,  "tcga_brca": 2, "liquid_asset": 2}

INPUT_TYPE = {"mnist": "image", "fashion_mnist": "image",
               "camelyon16": "image", "heart_disease": "tabular", "isic2019": "image",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "tabular"}

OUTPUT_TYPE = {"mnist": "discrete", "fashion_mnist": "discrete",
               "camelyon16": "discrete", "heart_disease": "discrete", "isic2019": "discrete",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "continuous"}

NB_CLIENTS = {"mnist": 6, "fashion_mnist": 10, "cifar10": 6, "camelyon16": 2, "heart_disease": 4, "isic2019": 6,
              "ixi": 3, "kits19": 6, "lidc_idri": 5, "tcga_brca": 6, "liquid_asset": 100, "synth": 2}

# MODELS = {"mnist": CNN_MNIST, "cifar10": CNN_CIFAR10}
# CRITERION = {"mnist": nn.CrossEntropyLoss(), "cifar10": nn.CrossEntropyLoss()}

MODELS = {"mnist": CNN_MNIST, "cifar10": GoogleNetTransferLearning, "heart_disease": HeartDiseaseRegression,
          "tcga_brca": TcgaRegression, "ixi": UNet, "liquid_asset": LiquidAssetRegression, "synth": SynthDataRegression}
CRITERION = {"mnist": nn.CrossEntropyLoss, "cifar10": nn.CrossEntropyLoss, "heart_disease": nn.BCELoss,
             "tcga_brca": CoxLoss, "ixi": DiceLoss, "liquid_asset": L1WeightedAccuracyLoss, "synth": MSELoss}
METRIC =  {"mnist": accuracy, "cifar10": accuracy, "heart_disease": auc, "tcga_brca": c_index, "ixi": dice,
           "liquid_asset": l1_accuracy, "synth": mse_accuracy}

### TODO : Il y a un problème avec la CoxLoss.

TRANSFORM_TRAIN = {"mnist": TRANSFORM_MNIST, "cifar10": TRANSFORM_TRAIN_CIFAR10,
                   "liquid_asset": None}
TRANSFORM_TEST= {"mnist": TRANSFORM_MNIST, "cifar10": TRANSFORM_TEST_CIFAR10,
                 "liquid_asset": None}
# Heart disease
# 0.7575019001960754 and parameters: {'step_size': 0.0030825079555064244, 'weight_decay': 0.01, 'scheduler_gamma': 0.7748947717692919}. Best is trial 6 with value: 0.7575019001960754.
# --- nb_of_communication: 25 - inner_epochs 50 ---
# TCGA BRCA
# {'step_size': 0.09905335662224482, 'weight_decay': 0.0001, 'scheduler_gamma': 0.99720911830093}. Best is trial 14 with value: 0.7716109471962976.
# --- nb_of_communication: 25 - inner_epochs 50 ---

STEP_SIZE = {"mnist": 0.01, "cifar10": 0.001, "tcga_brca": .09905335662224482, "heart_disease": 0.0030825079555064244, "ixi": 0.001,
             "liquid_asset": 0.1, "synth": 0.1}
WEIGHT_DECAY = {"mnist": 0.01, "cifar10": 0.001, "tcga_brca": .0001, "heart_disease": 0.01, "ixi": 0.001,
             "liquid_asset": 0.1, "synth": 0.1}
BATCH_SIZE = {"mnist": 256, "cifar10": 256, "tcga_brca": 8, "heart_disease": 4, "ixi": 32, "liquid_asset": 32,
              "synth": 64}
MOMENTUM = {"mnist": 0., "cifar10": 0.9, "tcga_brca": 0.95, "heart_disease": 0, "ixi": 0.95, "liquid_asset": 0.,
            "synth": 0}
SCHEDULER_PARAMS = {"mnist": (10, 2/3), "cifar10": (10, 2/3), "tcga_brca": (15, 0.99720911830093), "heart_disease": (15, 0.7748947717692919),
                    "ixi": (10, 2/3), "liquid_asset": (10, 2/3), "synth": (5, 2/3)}







