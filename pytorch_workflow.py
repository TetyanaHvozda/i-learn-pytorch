import torch
from torch import nn # PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

what_were_covering = {1: "data (prepare and load)",
                      2: "build model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluating a model (inference)",
                      5: "saving and loading a model",
                      6: "putting it all together"}

print(what_were_covering)

print(torch.__version__)

# Data (prepare and load) - linear regression formula
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])