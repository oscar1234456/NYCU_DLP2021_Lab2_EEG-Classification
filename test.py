import torch

print(torch.__version__)
print(torch.cuda.is_available())
torch.cuda.is_initialized()