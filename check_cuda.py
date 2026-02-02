#!/usr/bin/env python
"""Quick CUDA check script."""
import torch
import sys

print("="*60)
print("CUDA Status Check")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA is NOT available! Model will run on CPU (very slow)")
print("="*60)

# Test if we can create a tensor on GPU
if torch.cuda.is_available():
    try:
        x = torch.randn(100, 100).cuda()
        print("✓ Successfully created tensor on GPU")
        print(f"  Tensor device: {x.device}")
    except Exception as e:
        print(f"✗ Error creating tensor on GPU: {e}")
else:
    print("⚠ Cannot test GPU tensor creation - CUDA not available")
