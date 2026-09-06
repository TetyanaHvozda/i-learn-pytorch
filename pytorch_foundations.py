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