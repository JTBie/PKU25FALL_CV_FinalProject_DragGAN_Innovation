import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000).cuda()
    print("Test tensor on:", x.device)
    print("Using GPU: OK")
else:
    print("WARNING: Using CPU!")






