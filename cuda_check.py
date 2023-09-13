import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    print("GPU device name:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

