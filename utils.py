## Imports
import os
import random

import numpy as np
import torch

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if multi-GPU
    torch.backends.cudnn.deterministic=True # only applies to CUDA convolution operations
    torch.backends.cudnn.benchmark = False
    # usually CuDNN has heuristics as to which algorithm to pick. 
    # cudnn.benchmark benchmarks several algorithms and picks the fastest, which is often helpful 
    # if your input shapes are fixed and not changing a lot during training. However, this means it 
    # may pick a different algorithm even when the deterministic flag is set.
    # As such it is good practice to turn off cudnn.benchmark when turning on cudnn.deterministic

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')