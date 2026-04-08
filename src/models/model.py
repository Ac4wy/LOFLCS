from src.models.cifar_alexnet import CIFAR10_AlexNet
from src.models.mnist_cnn import Mnist_CNN
from src.utils.torch_utils import setup_seed
from src.models.cifar10_cnn import CIFAR10_CNN
from src.models.emnist_cnn import EMNIST_CNN
from src.models.fmnist_cnn import FMnist_CNN

import torch
import torch.nn as nn
import numpy as np


def choose_model(options):
    model_name = str(options['model_name']).lower()
    torch.manual_seed(2001)
    if model_name == 'mnist_cnn':

        # for name, param in Mnist_CNN().named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        return Mnist_CNN()
    elif model_name == 'alex':
        return CIFAR10_AlexNet()
    elif model_name == 'fmnist_cnn':
        return FMnist_CNN()
    elif model_name == 'emnist_cnn':
        return EMNIST_CNN()
    elif model_name == 'cifar10_alexnet':
        return CIFAR10_AlexNet()


