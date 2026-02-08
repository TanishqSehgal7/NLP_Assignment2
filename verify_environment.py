#!/usr/bin/env python3
"""
Environment verification script for CNN Assignment
"""

def check_environment():
    print("=" * 60)
    print("CNN ASSIGNMENT ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    try:
        # Core libraries
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import accuracy_score
        from PIL import Image
        import cv2
        import os
        
        print("✓ All required libraries imported successfully!")
        
        # Version information
        print(f"\nLibrary Versions:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Torchvision: {torchvision.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Pandas: {pd.__version__}")
        print(f"  OpenCV: {cv2.__version__}")
        
        # Device information
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice Information:")
        print(f"  Device: {device}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
            print(f"  Current CUDA Device: {torch.cuda.current_device()}")
        
        # Check data directory
        data_dir = "data/food-101-tiny"
        if os.path.exists(data_dir):
            print(f"\n✓ Data directory found: {data_dir}")
            train_dir = os.path.join(data_dir, "train")
            valid_dir = os.path.join(data_dir, "valid")
            
            if os.path.exists(train_dir):
                classes = os.listdir(train_dir)
                print(f"  Training classes: {len(classes)}")
                print(f"  Classes: {classes}")
            
            if os.path.exists(valid_dir):
                val_classes = os.listdir(valid_dir)
                print(f"  Validation classes: {len(val_classes)}")
        else:
            print(f"\n⚠ Data directory not found: {data_dir}")
            print("  Please ensure your dataset is in the correct location")
        
        print("\n" + "=" * 60)
        print("ENVIRONMENT SETUP COMPLETE!")
        print("You can now run your CNN assignment notebook.")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your environment setup.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_environment()