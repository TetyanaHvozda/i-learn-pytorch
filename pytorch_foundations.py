import torch

# print(torch.__version__)

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using Apple Silicon GPU (MPS)")
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

# Create a range of tensors and tensor-like
one_to_ten = torch.arange(start=0, end=11, step=1)
print(one_to_ten)

ten_zeroes = torch.zeros_like(input=one_to_ten)
print(ten_zeroes)

# Tensor datatypes
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None,
                               device="mps",
                               requires_grad=False) # track gradients
print(float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor.dtype)

# Getting information from a tensor
tensor = torch.rand(3, 4)
print(f"Data type of tensor: {tensor.dtype}")
print(f"hape of a tensor: {tensor.shape}")
print(f"Device of a tensor: {tensor.device}")

# Addition
tensor = torch.tensor([1, 2, 3])
tensor = tensor + 100
print(tensor)

# Multiply
tensor = tensor * 10
print(tensor)

# Subtract
tensor = tensor - 10
print(tensor)

# PyTorch in-built functions
tensor = torch.mul(tensor, 10)
print(tensor)

tensor = torch.add(tensor, 10)
print(tensor)

# Matrix multiplication
tensor = torch.tensor([1, 2, 3])
print(tensor * tensor)
print(torch.matmul(tensor, tensor))

# inner dimensions must be the same shape, output: outer dimensions
print(torch.matmul(torch.rand(3, 2), torch.rand(2, 3)))

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
print(tensor_B)
print(tensor_B.T)

# Tensor aggregation
x = torch.arange(0, 100, 10)
print(x.dtype)
print(torch.min(x))
print(x.min())

print(torch.max(x))
print(x.max())

#torch.mean requires a tensor of float32 datatype
print(torch.mean(x.type(torch.float32)))
print(x.type(torch.float32).mean())

print(torch.sum(x))
print(x.sum())

# Finding the positional min and max
print(x.argmin()) # index value
print(x[0]) # at index 0

print(x.argmax())
print(x[9])

# Reshaping, viewing, squeezing and stacking
x = torch.arange(1., 10.)
print(x)
print(x.shape)

x_reshaped = x.reshape(1, 9)
print(x_reshaped)
print(x_reshaped.shape)

x_reshaped = x.reshape(9, 1)
print(x_reshaped)
print(x_reshaped.shape)

z = x.view(1, 9)
print(z)
print(z.shape)

# changing z changes x
z[0, 1] = 5
print(z)
print(x)

x_stacked = torch.stack([x, x, x, x], dim=0)
print(x)
print(x.shape)
print(x_stacked)
print(x_stacked.shape)

x_stacked = torch.stack([x, x, x, x], dim=1)
print(x)
print(x.shape)
print(x_stacked)
print(x_stacked.shape)

# torch.squeeze() - removes all single dimensions from a target tensor
x_reshaped = x.reshape(1, 9)
print(x_reshaped)
print(x_reshaped.shape)
x_squeezed = x_reshaped.squeeze()
print(x_squeezed)
print(x_reshaped.squeeze().shape)

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed)
print(x_unsqueezed.shape)

# torch.permute - rearanges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # hoght, width, colour_channels
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(x_original.shape)
print(x_permuted.shape)

# indexing
x_original[0, 0, 0] = 728218
print(x_original[0, 0, 0])
print(x_permuted[0, 0, 0])

x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print(x.shape)

print(x[0]) # 0th dimension
print(x[0][0]) # also x[0, 0], 1st dimension
print(x[0][0][0]) # 2nd dimension

print(x[:, 0])
# all values in the 0th and 1st dimensions but only index 1 of second dimension
print(x[:, :, 1])
# all values of the 0 dim but only the 1 index value of 1st and 2nd dimension
print(x[:, 1, 1])
# index 0 of 0th and 1st dimension and all alues of 2nd dimension
print(x[0, 0, :])

# Index on x to return 9
print(x[0][2][2])
# Index on x to return 3, 6, 9
print(x[:, :, 2])

# numpy array to tensor
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # a new tensor in memory
print(array) 
print(tensor)
print(array.dtype) # default numpy dtype is float64
print(tensor.dtype)
print(tensor.type(torch.float32).dtype) # convert to float32

array = array + 1 
print(array)
print(tensor)

# tensor to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor.dtype)
print(numpy_tensor.dtype)

# change the tensor, what happens to numpy
tensor = tensor + 1
print(tensor)
print(numpy_tensor)

# reproducability (take random out of random)
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED) # works only for one block of code
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)