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

INPUT_TYPE = {"mnist": "image", "fashion_mnist": "image",
               "camelyon16": "image", "heart_disease": "tabular", "isic2019": "image",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "tabular"}

OUTPUT_TYPE = {"mnist": "discrete", "fashion_mnist": "discrete",
               "camelyon16": "discrete", "heart_disease": "discrete", "isic2019": "discrete",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "continuous"}

NB_CLIENTS = {"mnist": 6, "fashion_mnist": 8, "cifar10": 8, "camelyon16": 2, "heart_disease": 4, "isic2019": 6,
              "ixi": 3, "kits19": 6, "lidc_idri": 5, "tcga_brca": 6, "liquid_asset": 100, "synth": 2,
              "exam_llm": 3}

MODELS = {"mnist": CNN_MNIST, "cifar10": GoogleNetTransferLearning, "heart_disease": HeartDiseaseRegression,
          "tcga_brca": TcgaRegression, "ixi": UNet, "liquid_asset": LiquidAssetRegression, "synth": SynthDataRegression,
          "exam_llm": None}
CRITERION = {"mnist": nn.CrossEntropyLoss, "cifar10": nn.CrossEntropyLoss, "heart_disease": nn.BCELoss,
             "tcga_brca": CoxLoss, "ixi": DiceLoss, "liquid_asset": L1WeightedAccuracyLoss, "synth": MSELoss,
             "exam_llm": nn.CrossEntropyLoss}
METRIC =  {"mnist": accuracy, "cifar10": accuracy, "heart_disease": auc, "tcga_brca": c_index, "ixi": dice,
           "liquid_asset": l1_accuracy, "synth": mse_accuracy,
           "exam_llm": accuracy}

CHECKPOINT = "distilbert-base-uncased" #"bert-base-uncased"

### TODO : Il y a un problème avec la CoxLoss.

TRANSFORM_TRAIN = {"mnist": TRANSFORM_MNIST, "cifar10": TRANSFORM_TRAIN_CIFAR10,
                   "liquid_asset": None}
TRANSFORM_TEST= {"mnist": TRANSFORM_MNIST, "cifar10": TRANSFORM_TEST_CIFAR10,
                 "liquid_asset": None}

STEP_SIZE = {"mnist": 0.045, "cifar10": 0.001, "tcga_brca": .015, "heart_disease": 0.001,
             "ixi": 0.01, "liquid_asset": 0.001, "synth": 0.1, "exam_llm": 10**-3}
WEIGHT_DECAY = {"mnist": 0, "cifar10": 0.001, "tcga_brca": .001, "heart_disease": 0,
                "ixi": 0.0001, "liquid_asset": 0.1, "synth": 0.1, "exam_llm": 10**-4}
BATCH_SIZE = {"mnist": 256, "cifar10": 256, "tcga_brca": 8, "heart_disease": 4, "ixi": 32, "liquid_asset": 32,
              "synth": 64, "exam_llm": 8}
MOMENTUM = {"mnist": 0., "cifar10": 0.95, "tcga_brca": 0, "heart_disease": 0, "ixi": 0.95,
            "liquid_asset": 0.95, "synth": 0, "exam_llm": 0.95}
SCHEDULER_PARAMS = {"mnist": (15, 0.92), "cifar10": (15, 1), "tcga_brca": (50, 0.609), "heart_disease": (15, 0.554),
                    "ixi": (15, 0.75), "liquid_asset": (15, 0.75), "synth": (15, 2/3), "exam_llm": (15, 2/3)}

# MNIST
# Best trial config:  {'step_size': 0.04500608326765825, 'weight_decay': 0, 'scheduler_gamma': 0.9254090508955067}
# Best trial final validation accuracy:  0.9909150004386902
# IXI
# Best trial config:  {'step_size': 0.019674439294335523, 'weight_decay': 0.001, 'scheduler_gamma': 0.6762453175771046}
# Best trial final validation accuracy:  0.9775788187980652






