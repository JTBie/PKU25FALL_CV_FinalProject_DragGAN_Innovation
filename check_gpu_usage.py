#!/usr/bin/env python
"""Check if DragGAN is using GPU or CPU."""

import torch
import sys
import os

print("=" * 70)
print("GPU/CUDA Usage Check")
print("=" * 70)

# Check PyTorch CUDA
print(f"\n[1] PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    Device Count: {torch.cuda.device_count()}")
    print(f"    Current Device: {torch.cuda.current_device()}")
    print(f"    Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor on GPU
    x = torch.randn(1000, 1000).cuda()
    print(f"    Test Tensor Device: {x.device}")
    print(f"    Test Tensor Location: {'GPU' if x.device.type == 'cuda' else 'CPU'}")
else:
    print("    [WARNING] CUDA is not available! Using CPU.")

# Check CUDA extensions
print("\n[2] CUDA Extensions Status:")
try:
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
    
    # Check if extensions are actually using CUDA
    if torch.cuda.is_available():
        x = torch.randn(1, 3, 64, 64).cuda()
        b = torch.randn(3).cuda()
        
        # Test bias_act
        try:
            result = bias_act.bias_act(x, b)
            if result.device.type == 'cuda':
                print("    [OK] bias_act: Using GPU")
            else:
                print("    [WARNING] bias_act: Using CPU (fallback)")
        except Exception as e:
            print(f"    [ERROR] bias_act: {e}")
        
        # Test upfirdn2d
        try:
            f = torch.ones([3, 3], dtype=torch.float32).cuda()
            result = upfirdn2d.upfirdn2d(x, f)
            if result.device.type == 'cuda':
                print("    [OK] upfirdn2d: Using GPU")
            else:
                print("    [WARNING] upfirdn2d: Using CPU (fallback)")
        except Exception as e:
            print(f"    [ERROR] upfirdn2d: {e}")
    else:
        print("    [SKIP] CUDA not available, cannot test extensions")
        
except Exception as e:
    print(f"    [ERROR] Failed to import extensions: {e}")

# Check Renderer device
print("\n[3] Checking Renderer Configuration:")
try:
    from viz.renderer import Renderer
    print("    Renderer class imported successfully")
    # Note: Renderer device is set when instantiated
except Exception as e:
    print(f"    [ERROR] Failed to import Renderer: {e}")

# Check environment
print("\n[4] Environment Check:")
print(f"    CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"    CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"    CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Check if there are any CPU fallback warnings
print("\n[5] Checking for CPU Fallback Warnings:")
print("    (Check the program output for 'Falling back to CPU' messages)")

print("\n" + "=" * 70)
if torch.cuda.is_available():
    print("[SUCCESS] CUDA is available and should be used")
else:
    print("[WARNING] CUDA is NOT available - program will use CPU (very slow!)")
print("=" * 70)






