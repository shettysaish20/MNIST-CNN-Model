import torch
import sys
from mnist_working_model import MNIST_CNN_Second

def check_model():
    model = MNIST_CNN_Second()
    
    # Check 1: Total Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    if total_params == 0:
        print('❌ Error: Model has no parameters')
        sys.exit(1)
    print(f'✅ Total parameters: {total_params}')
    
    # Check 2: Batch Normalization
    has_batchnorm = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    if not has_batchnorm:
        print('❌ Error: Model does not use Batch Normalization')
        sys.exit(1)
    print('✅ Model uses Batch Normalization')
    
    # Check 3: Dropout
    has_dropout = any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    if not has_dropout:
        print('❌ Error: Model does not use Dropout')
        sys.exit(1)
    print('✅ Model uses Dropout')
    
    # Check 4: Fully Connected Layer or GAP
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())
    if not (has_fc or has_gap):
        print('❌ Error: Model does not use either Fully Connected Layer or Global Average Pooling')
        sys.exit(1)
    print('✅ Model uses FC Layer or GAP')

if __name__ == '__main__':
    check_model() 