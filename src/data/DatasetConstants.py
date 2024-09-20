"""Created by Constantin Philippenko, 12th May 2022."""

import torchvision
import torch.nn as nn
from torchvision.transforms import transforms

from src.optim.CustomLoss import DiceLoss, CoxLoss
from src.optim.Metric import dice, auc, c_index, accuracy
from src.optim.nn.Nets import LinearRegression, LogisticRegression, TcgaRegression, CNN_CIFAR10, CNN_MNIST, \
    HeartDiseaseRegression, ResNet50TransferLearning
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

TRANSFORM_TEST_CIFAR10 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

MAX_NB_POINTS = 20000

NB_LABELS = {"mnist": 10, "fashion_mnist": 10, "cifar10": 10,
             "camelyon16": 2,  "heart_disease": 2, "isic2019": 8,
             "ixi": None, "kits19": None, "lidc_idri": None,  "tcga_brca": 2}

INPUT_TYPE = {"mnist": "image", "fashion_mnist": "image",
               "camelyon16": "image", "heart_disease": "tabular", "isic2019": "image",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "tabular"}

OUTPUT_TYPE = {"mnist": "discrete", "fashion_mnist": "discrete",
               "camelyon16": "discrete", "heart_disease": "discrete", "isic2019": "discrete",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "continuous"}

NB_CLIENTS = {"mnist": 12, "fashion_mnist": 10, "cifar10": 7, "camelyon16": 2, "heart_disease": 4, "isic2019": 6,
              "ixi": 3, "kits19": 6, "lidc_idri": 5, "tcga_brca": 6}

# MODELS = {"mnist": CNN_MNIST, "cifar10": CNN_CIFAR10}
# CRITERION = {"mnist": nn.CrossEntropyLoss(), "cifar10": nn.CrossEntropyLoss()}

MODELS = {"mnist": CNN_MNIST, "cifar10": ResNet50TransferLearning, "heart_disease": HeartDiseaseRegression,
          "tcga_brca": TcgaRegression, "ixi": UNet}
CRITERION = {"mnist": nn.CrossEntropyLoss, "cifar10": nn.CrossEntropyLoss, "heart_disease": nn.BCELoss,
             "tcga_brca": CoxLoss, "ixi": DiceLoss}
METRIC =  {"mnist": accuracy, "cifar10": accuracy, "heart_disease": auc, "tcga_brca": c_index, "ixi": dice}

### TODO : Il y a un problème avec la CoxLoss.

TRANSFORM_TRAIN = {"mnist": TRANSFORM_MNIST, "cifar10": TRANSFORM_TRAIN_CIFAR10}
TRANSFORM_TEST= {"mnist": TRANSFORM_MNIST, "cifar10": TRANSFORM_TEST_CIFAR10}
STEP_SIZE = {"mnist": 0.1, "cifar10": 0.001, "tcga_brca": .01, "heart_disease": 0.01, "ixi": 0.001}
BATCH_SIZE = {"mnist": 256, "cifar10": 256, "tcga_brca": 4, "heart_disease": 4, "ixi": 128}
MOMENTUM = {"mnist": 0.9, "cifar10": 0.95, "tcga_brca": 0, "heart_disease": 0, "ixi": 0.95}







