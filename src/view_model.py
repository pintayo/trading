import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.model_config import *

model = torch.load(SIMPLE_MODEL_PATH)
print(model)