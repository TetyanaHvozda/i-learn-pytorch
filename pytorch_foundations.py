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