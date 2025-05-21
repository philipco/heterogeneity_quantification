import torchvision
import torch.nn as nn
from torch.nn import MSELoss
from torchvision.transforms import transforms

from src.optim.CustomLoss import DiceLoss, CoxLoss, L1WeightedAccuracyLoss
from src.optim.Metric import dice, auc, c_index, accuracy, l1_accuracy, mse_accuracy
from src.optim.nn.Nets import TcgaRegression, CNN_MNIST, HeartDiseaseRegression, LiquidAssetRegression, \
    Synth2ClientsRegression, Synth100ClientsRegression, LeNet
from src.optim.nn.Unet import UNet

PCA_NB_COMPONENTS = 16

TRANSFORM_MNIST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

TRANSFORM_TRAIN_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),         
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

TRANSFORM_TEST_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

MODELS = {"mnist": CNN_MNIST, "mnist_iid": CNN_MNIST, "cifar10": LeNet, "cifar10_iid": LeNet,
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

CHECKPOINT = "distilbert-base-uncased"

TRANSFORM_TRAIN = {"mnist": TRANSFORM_MNIST, "mnist_iid": TRANSFORM_MNIST, "cifar10": TRANSFORM_TRAIN_CIFAR10, "cifar10_iid": TRANSFORM_TRAIN_CIFAR10,
                   "liquid_asset": None}
TRANSFORM_TEST= {"mnist": TRANSFORM_MNIST, "mnist_iid": TRANSFORM_MNIST, "cifar10": TRANSFORM_TEST_CIFAR10, "cifar10_iid": TRANSFORM_TEST_CIFAR10,
                 "liquid_asset": None}

NB_CLIENTS = {"mnist": 20, "mnist_iid": 20, "fashion_mnist": 8, "cifar10": 20, "cifar10_iid": 20, "camelyon16": 2, "heart_disease": 4, "isic2019": 6,
              "ixi": 3, "kits19": 6, "lidc_idri": 5, "tcga_brca": 6, "liquid_asset": 100, "synth": 20,
              "synth_complex": 20,
              "exam_llm": 3}
STEP_SIZE = {"mnist": 0.1, "mnist_iid": 0.1, "cifar10": 0.1, "cifar10_iid": 0.1, "tcga_brca": .015, "heart_disease": 0.05,
             "ixi": 0.01, "liquid_asset": 0.01, "synth": None, "synth_complex": None,
             "exam_llm": 0.001}
WEIGHT_DECAY = {"mnist": 5*10**-4, "mnist_iid": 5*10**-4, "cifar10": 5*10**-4, "cifar10_iid": 5*10**-4, "tcga_brca": 0, "heart_disease": 0,
                "ixi": 5*10**-4, "liquid_asset": 5*10**-4, "synth": 0, "synth_complex": 0, "exam_llm": 5*10**-4}
BATCH_SIZE = {"mnist": 64, "mnist_iid": 64, "cifar10": 64, "cifar10_iid": 64, "tcga_brca": 8, "heart_disease": 1,
              "ixi": 8, "liquid_asset": 8, "synth": 1, "synth_complex": 1, "exam_llm": 8}
MOMENTUM = {"mnist": 0., "mnist_iid": 0., "cifar10": 0.9, "cifar10_iid": 0.9, "tcga_brca": 0, "heart_disease": 0,
            "ixi": 0.9, "liquid_asset": 0.9, "synth": 0, "synth_complex":0, "exam_llm": 0.9}
SCHEDULER_PARAMS = {"mnist": (10, 0.1), "mnist_iid": (10, 0.1), "cifar10": (10, 0.1), "cifar10_iid": (10, 0.1), "tcga_brca": (50, 0.609),
                    "heart_disease": (5, 0.1),
                    "ixi": (10, 0.1), "liquid_asset": (1, 0.9), "synth": (1, 1), "synth_complex": (1, 1),
                    "exam_llm": (10, 0.1)}
NB_EPOCHS = {"mnist": 50, "mnist_iid": 50, "cifar10": 50, "cifar10_iid": 50, "tcga_brca": 50, "heart_disease": 25,
            "ixi": 10, "liquid_asset": 50, "synth": 100, "synth_complex": 200, "exam_llm": 4}






