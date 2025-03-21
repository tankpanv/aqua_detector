# test_ctypes.py
import ctypes
import torch

print("ctypes模块位置:", ctypes.__file__)
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())