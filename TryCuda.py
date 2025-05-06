import torch

print("CUDA 可用性：", torch.cuda.is_available())
print("当前设备：", torch.cuda.get_device_name(0))
print("PyTorch 使用的 CUDA 版本：", torch.version.cuda)
