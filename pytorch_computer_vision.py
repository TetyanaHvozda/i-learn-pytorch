# torchvision - base computer vision library
# torchvision.datasets
# torchvision.models
# torchvision.transforms - functions for maintaining images (from images to numbers)
#torch.utils.data.Dataset
# torch.utils.data.DataLoader - Python iterable over a dataset

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

# 1. Dataset Fashion MNIST
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor,
    target_transform=None
)

len(train_data), len(test_data) # 60000, 10000

# see the first training example
image, label = train_data[0]
image, label

class_names = train_data.classes
class_names

class_to_idx = train_data.class_to_idx
class_to_idx

train_data.targets

image.shape, label
