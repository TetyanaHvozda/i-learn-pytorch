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


