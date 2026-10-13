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

# Visualizing
import matplotlib.pyplot as plt
image, label = train_data[0]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze())
plt.title(label)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)

# randomly look at the data
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    print(random_idx)
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[label])
    plt.axis(False)

