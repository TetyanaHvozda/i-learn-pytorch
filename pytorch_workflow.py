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

# split data into test and train
# train - learn patterns, 60-80%
# validation - tune model patterns, 10-20% (optional)
# test - see if the model is ready for unseen data, 10-20%
train_split = int(0.8 * len(X))
print(train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Visualize
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

#plot_predictions()
#plt.show()

# Build a PyTorch model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

# torch.optim -> optimizer
# def forward() -> All nn.Module subclasses require you to overwrite

# Check the content of the model
torch.manual_seed(42)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
print(model_0.state_dict())

# Making predictions
with torch.inference_mode(): # turns off gradient tracking
    y_preds = model_0(X_test)

print(y_preds)

# plot_predictions(predictions=y_preds)
# plt.show()

# Train model
# Loss function - measure how wrong is the prediction; lower better
# Optimizer - takes the loss into account and adjusts parameters (e.g. weights and bias) to improve the loss function
# for PyTorch: a training loop and a testing loop
print(list(model_0.parameters()))
print(model_0.state_dict())

# MAE (l1 loss)
loss_fn = nn.L1Loss()
# randomly adjusting mean values
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

## Training
# training loop steps and intuition
epochs = 100

for epoch in range(epochs):
    # set the model to training mode
    model_0.train() # sets all parameters that requires gradient

    # Forward pass
    y_pred = model_0(X_train)

    # Calculate the loss
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()

    # perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # step the optimizer (perform gradient descent)
    optimizer.step()
    ## Testing; turns off different settings in the model (dropuot, batch norm) not needed for evaluation
    model_0.eval() # turns off gradient tracking
    with torch.inference_mode():
    # with torch.no_grad(): in older code
        #1. forward pass
        test_pred = model_0(X_test)

        #2. calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())

with torch.inference_mode():
    y_preds_new = model_0(X_test)

# plot_predictions(predictions=y_preds_new)
# plt.show()


