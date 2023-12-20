"""Created by Constantin Philippenko, 12th May 2022."""
import torchvision
import torch.nn as nn

from src.optim.Nets import LinearRegression, CNNClassifier
from src.utils.UtilitiesPytorch import ReshapeTransform

NB_LABELS = {"mnist": 10, "fashion_mnist": 10,
             "camelyon16": 2,  "heart_disease": 2, "isic2019": 8,
             "ixi": None, "kits19": None, "lidc_idri": None,  "tcga_brca": 2}

INPUT_TYPE = {"mnist": "image", "fashion_mnist": "image",
               "camelyon16": "image", "heart_disease": "tabular", "isic2019": "image",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "tabular"}

OUTPUT_TYPE = {"mnist": "discrete", "fashion_mnist": "discrete",
               "camelyon16": "discrete", "heart_disease": "discrete", "isic2019": "discrete",
               "ixi": "image", "kits19": "image", "lidc_idri": "image", "tcga_brca": "continuous"}

NB_CLIENTS = {"mnist": 7, "fashion_mnist": 10, "camelyon16": 2, "heart_disease": 4, "isic2019": 6,
              "ixi": 3, "kits19": 6, "lidc_idri": 5, "tcga_brca": 6}

PCA_NB_COMPONENTS = 16

TRANSFORM_MIST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        # We reshape mnist to match with our neural network
        ReshapeTransform((-1,))

    ])

MODELS = {"cnn": CNNClassifier, "linear": LinearRegression}
CRITERION = {"crossentropy": nn.CrossEntropyLoss(), "linear": nn.MSELoss(), "binary": nn.BCELoss()}